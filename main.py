# Example Run Command (single GPU):
#   python main.py \
#       --train_csv /path/to/train.csv \
#       --valid_csv /path/to/valid.csv \
#       --output_dir ./multi_label_ckpt \
#       --epochs 3 \
#       --batch_size 4 \
#       --lr 2e-5 \
#       --max_length 256 \
#       --sample_checkpoint_interval 10000 \
#       --fp16 True \
#       --gradient_checkpointing True \
#       --run_inference False \
#       --class_weights True \
#       --early_stopping_patience 2 \
#       --tune_thresholds True
#
# If training crashes for some reason, re-running the same command automatically resumes
# from the last sample-based checkpoint, so there is no lose of training progress.
# If you set --tune_thresholds True, after training, we find optimal label thresholds
# from the validation set, use them for final metrics and for batch inference as well.
#
# For multi-GPU usage:
#   torchrun --nproc_per_node=4 main.py ...

import os
import re
import csv
import json
import math
import argparse
import logging
import fnmatch
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import numpy as np

# Hugging Face
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    set_seed,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict

# For advanced multi-label metrics
from sklearn.metrics import precision_recall_fscore_support

###############################################################################
# logging setup global
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


###############################################################################
# argument parsing
###############################################################################
def parse_args():
    """
    Additional arguments:
      --class_weights           : If True, compute label frequencies & apply Weighted BCEWithLogitsLoss.
      --early_stopping_patience : If > 0, sets EarlyStoppingCallback to stop training if metric stops improving.
      --tune_thresholds         : If True, after training, find label-wise thresholds on the validation set that maximize F1 per label.
    """
    parser = argparse.ArgumentParser(
        description="Train a multi-label hate-speech classifier with advanced features."
    )

    # Required data paths
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to training CSV with 'text' + 7 binary columns.",
    )
    parser.add_argument(
        "--valid_csv",
        type=str,
        required=True,
        help="Path to validation CSV with 'text' + 7 binary columns.",
    )

    # Output / model config
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./xlmr_hate_speech_ckpt",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Per-device batch size for training."
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization.",
    )

    # Logging / saving intervals
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log training metrics every X steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save model checkpoint every X steps (normal HF approach).",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Evaluate model every X steps."
    )
    parser.add_argument(
        "--sample_checkpoint_interval",
        type=int,
        default=10000,
        help="Save an additional disk checkpoint after every N training samples.",
    )
    parser.add_argument(
        "--auto_resume_from_latest",
        type=lambda x: x.lower() == "true",
        default=True,
        help="If True, automatically resume from the latest sample-based checkpoint on startup.",
    )

    # Training hyperparams
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1, help="LR warmup ratio."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay for AdamW."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients across N steps.",
    )

    # Mixed precision & gradient checkpointing
    parser.add_argument(
        "--fp16",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use mixed-precision FP16 training if True.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable gradient checkpointing if True.",
    )

    # Hugging Face Hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push final model to Hugging Face Hub if set.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hugging Face Model Hub ID (user/repo).",
    )

    # DeepSpeed & inference
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON (optional).",
    )
    parser.add_argument(
        "--run_inference",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Run inference code after training if True.",
    )

    # New improvement flags
    parser.add_argument(
        "--class_weights",
        type=lambda x: x.lower() == "true",
        default=False,
        help="If True, compute label frequencies and use weighted BCE loss.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="If > 0, use EarlyStoppingCallback with that patience.",
    )
    parser.add_argument(
        "--tune_thresholds",
        type=lambda x: x.lower() == "true",
        default=False,
        help="If True, find label-wise thresholds after training using the validation set.",
    )
    args = parser.parse_args()
    return args


###############################################################################
# csv data loading
###############################################################################
def preprocess_text(text: str) -> str:
    """
    Text preprocessing hook:
    remove URLs & lower-case the text
    """
    import re

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Lowercase
    text = text.lower()
    return text


def read_csv_to_dict(csv_path: str) -> List[Dict[str, Any]]:
    """
    Reads a CSV with columns:
      'text'
      'race', 'religion', 'gender', 'sexual_orientation', 'nationality',
      'disability', 'misc_hate'
    Each label is 0 or 1.

    Includes basic data validation checks for missing columns or invalid (non-binary) labels.
    Also calls a preprocess_text hook for demonstration.
    """
    categories = [
        "race",
        "religion",
        "gender",
        "sexual_orientation",
        "nationality",
        "disability",
        "misc_hate",
    ]

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Basic validation
        missing_cols = [cat for cat in categories if cat not in reader.fieldnames]
        if "text" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'text' column.")
        if missing_cols:
            raise ValueError(f"CSV is missing required label columns: {missing_cols}")

        for row_idx, row in enumerate(reader):
            raw_text_value = row["text"]
            # Preprocess
            text_value = preprocess_text(raw_text_value)

            labels = []
            for cat in categories:
                raw_val = row[cat]
                try:
                    label_val = int(raw_val)
                except ValueError:
                    raise ValueError(
                        f"Row {row_idx}: label '{cat}' must be 0 or 1, got '{raw_val}'."
                    )
                if label_val not in [0, 1]:
                    raise ValueError(
                        f"Row {row_idx}: label '{cat}' must be 0 or 1, got '{label_val}'."
                    )
                labels.append(label_val)

            sample_dict = {"text": text_value, "labels": labels}
            data.append(sample_dict)
    return data


def load_dataset_dict(train_csv: str, valid_csv: str) -> DatasetDict:
    logging.info(f"Loading train data from {train_csv} ...")
    train_data = read_csv_to_dict(train_csv)
    logging.info(f"Loading validation data from {valid_csv} ...")
    valid_data = read_csv_to_dict(valid_csv)

    train_ds = Dataset.from_list(train_data)
    valid_ds = Dataset.from_list(valid_data)

    ds_dict = DatasetDict({"train": train_ds, "validation": valid_ds})
    return ds_dict


###############################################################################
# tokenization
###############################################################################
def tokenize_batch(examples, tokenizer, max_length: int):
    """
    Tokenize the 'text' field, carry over 'labels'.
    """
    tokenized = tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=max_length
    )
    tokenized["labels"] = examples["labels"]
    return tokenized


###############################################################################
# class weights - for imbalance
###############################################################################
def compute_label_frequencies(dataset, num_labels=7):
    freqs = [0] * num_labels
    total = 0
    for sample in dataset:
        labels = sample["labels"]  # like example [0,1,0,1, ...]
        for i, val in enumerate(labels):
            freqs[i] += val
        total += 1
    return freqs, total


def compute_class_weights(freqs, total):
    """
    If class i has freq f_i, you can compute weights in many ways.
    We'll do average_freq / freq_i.
    """
    num_labels = len(freqs)
    avg = float(sum(freqs)) / num_labels
    weights = []
    for f in freqs:
        if f == 0:
            w = 2.0 * avg
        else:
            w = avg / float(f)
        weights.append(w)
    return weights


###############################################################################
# Per-label metrics + Threshold tuning
###############################################################################
def evaluate_with_thresholds(
    probabilities: np.ndarray,
    references: np.ndarray,
    thresholds: List[float],
    categories: List[str],
):
    """
    Given a [batch, 7] probability array and a list of label-specific thresholds,
    we binarize each label using thresholds[label].
    Then compute micro/macro metrics, plus per-label P/R/F1.
    """
    binary_preds = np.zeros(probabilities.shape, dtype=int)
    for i in range(probabilities.shape[1]):  # for each label
        binary_preds[:, i] = (probabilities[:, i] >= thresholds[i]).astype(int)

    # Micro
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        references, binary_preds, average="micro", zero_division=0
    )
    # Macro
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        references, binary_preds, average="macro", zero_division=0
    )
    # Per-label
    p_per_label, r_per_label, f_per_label, _ = precision_recall_fscore_support(
        references, binary_preds, average=None, zero_division=0
    )
    # Subset accuracy
    subset_acc = np.mean(np.all(binary_preds == references, axis=1))

    metrics = {
        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "f1_micro": float(f_micro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "subset_accuracy": float(subset_acc),
    }
    for i, cat in enumerate(categories):
        metrics[f"precision_{cat}"] = float(p_per_label[i])
        metrics[f"recall_{cat}"] = float(r_per_label[i])
        metrics[f"f1_{cat}"] = float(f_per_label[i])
    return metrics


def find_optimal_thresholds(
    probabilities: np.ndarray, references: np.ndarray, categories: List[str]
):
    """
    For each label, search thresholds from 0.0 to 1.0 in steps of 0.01
    to find the threshold that maximizes F1 for that label individually.
    Return a list of 7 thresholds.
    - expensive for larger datasets, but it's a straightforward approach
    """
    num_labels = probabilities.shape[1]
    best_thresholds = [0.5] * num_labels
    logging.info(
        "[ThresholdTuning] Searching thresholds for each label from 0.0 to 1.0 (step=0.01) - May take time."
    )

    for label_idx in range(num_labels):
        best_thr = 0.5
        best_f1 = 0.0
        # A simple linear scan
        for thr in np.arange(0.0, 1.01, 0.01):
            preds_label = (probabilities[:, label_idx] >= thr).astype(int)
            refs_label = references[:, label_idx]
            p, r, f, _ = precision_recall_fscore_support(
                refs_label, preds_label, average="binary", zero_division=0
            )
            if f > best_f1:
                best_f1 = f
                best_thr = thr
        best_thresholds[label_idx] = best_thr
        logging.info(
            f"[ThresholdTuning] For label {categories[label_idx]}, best_f1={best_f1:.4f} at threshold={best_thr:.2f}"
        )
    return best_thresholds


def compute_multilabel_metrics_per_label(
    predictions, references, categories, label_thresholds=None
):
    """
    predictions: shape [batch, 7] (logits)
    references : shape [batch, 7] (0 or 1)
    If label_thresholds is None, default to 0.5 for all
    Otherwise, use the label_thresholds for binarizing each label
    """
    sigmoid = nn.Sigmoid()
    preds_tensor = torch.tensor(predictions)
    probs = sigmoid(preds_tensor).numpy()  # shape [batch, 7]

    if label_thresholds is None:
        label_thresholds = [0.5] * len(categories)

    metrics = evaluate_with_thresholds(probs, references, label_thresholds, categories)
    return metrics


###############################################################################
# custom trainer for class weighted loss
###############################################################################
class CustomWeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None  # will be a tensor like [w1, w2, ... w7]

    def set_class_weights(self, weights: torch.Tensor):
        self.class_weights = weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        new_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**new_inputs)
        logits = outputs.logits
        if labels is not None:
            labels = labels.float()
            if self.class_weights is not None:
                loss_fct = nn.BCEWithLogitsLoss(
                    pos_weight=self.class_weights.to(logits.device)
                )
            else:
                loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss
        else:
            return (outputs.loss, outputs) if return_outputs else outputs.loss


###############################################################################
# sample based check-pointing callback
###############################################################################
class SampleCheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_interval=10000, output_dir="./checkpoints"):
        super().__init__()
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = output_dir
        self.samples_seen = 0
        self.next_save_threshold = self.checkpoint_interval

    def on_train_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        logging.info(
            f"[SampleCheckpointCallback] Will save a disk checkpoint every {self.checkpoint_interval} samples."
        )

    def on_train_batch_end(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        trainer = kwargs.get("trainer", None)
        batch = kwargs.get("batch", None)

        if batch is not None and isinstance(batch, dict) and "input_ids" in batch:
            current_batch_size = batch["input_ids"].size(0)
        else:
            current_batch_size = 1

        self.samples_seen += current_batch_size

        if self.samples_seen >= self.next_save_threshold:
            logging.info(
                f"[SampleCheckpointCallback] Reached sample {self.samples_seen}. Saving checkpoint."
            )
            if trainer is not None:
                checkpoint_dir = os.path.join(
                    self.output_dir, f"checkpoint-sample-{self.samples_seen}"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                trainer.save_model(checkpoint_dir)
                trainer.save_state()
                logging.info(
                    f"[SampleCheckpointCallback] Disk checkpoint saved in {checkpoint_dir}."
                )
            else:
                logging.warning(
                    "[SampleCheckpointCallback] No 'trainer' in kwargs; cannot save checkpoint now."
                )

            self.next_save_threshold += self.checkpoint_interval

        return control


###############################################################################
# auto resume logic/feature
###############################################################################
def find_latest_sample_checkpoint(base_dir: str) -> Optional[str]:
    if not os.path.isdir(base_dir):
        return None

    max_checkpoint_dir = None
    max_samples = -1
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            match = re.match(r"checkpoint-sample-(\d+)", entry.name)
            if match:
                samples = int(match.group(1))
                if samples > max_samples:
                    max_samples = samples
                    max_checkpoint_dir = os.path.join(base_dir, entry.name)

    if max_checkpoint_dir is not None:
        logging.info(
            f"[AutoResume] Found sample-based checkpoint: {max_checkpoint_dir} (samples={max_samples})"
        )
    else:
        logging.info("[AutoResume] No sample-based checkpoints found in output_dir.")

    return max_checkpoint_dir


###############################################################################
# batch inference for large-scale usage
###############################################################################
def run_inference_in_batches(
    model_path: str, texts: List[str], batch_size: int = 32, max_length: int = 256
) -> np.ndarray:
    """
    For large-scale usage:
      - Load model + tokenizer once.
      - Process 'texts' in batches of 'batch_size'.
      - Return probabilities as a 2D numpy array [num_texts, 7].
    """
    logging.info(f"[BATCH INFERENCE] Loading model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=7, problem_type="multi_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_probs = []
    sigmoid = nn.Sigmoid()

    # Batching
    start_idx = 0
    while start_idx < len(texts):
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            logits = model(**encodings).logits  # shape [batch_size, 7]
            probs = sigmoid(logits)  # shape [batch_size, 7]
        all_probs.append(probs.cpu().numpy())
        start_idx = end_idx

    all_probs = np.concatenate(all_probs, axis=0)  # shape [num_texts, 7]
    return all_probs


def run_demo_inference_single_texts(
    model_path: str,
    texts: List[str],
    max_length: int,
    thresholds: Optional[List[float]] = None,
):
    """
    Demo function that calls run_inference_in_batches with batch_size=1 for each text,
    but you can do them in a single call as well. If thresholds is provided,
    we use them for final classification.
    """
    if thresholds is None:
        thresholds = [0.5] * 7  # fallback
    category_list = [
        "race",
        "religion",
        "gender",
        "sexual_orientation",
        "nationality",
        "disability",
        "misc_hate",
    ]
    # Do a single batch inference call for all texts
    probs_array = run_inference_in_batches(
        model_path, texts, batch_size=8, max_length=max_length
    )
    # Interpret results
    for text, probs in zip(texts, probs_array):
        bin_preds = [(1 if p >= thr else 0) for p, thr in zip(probs, thresholds)]
        prob_strs = [f"{p:.3f}" for p in probs]
        print("----------------------------------------------------------------")
        print(f"Text: {text}")
        print("Category Probabilities (and labels at thresholds):")
        for cat, p, b, thr in zip(category_list, prob_strs, bin_preds, thresholds):
            print(f"  - {cat}: prob={p} thr={thr:.2f} => label={b}")

    print("===================================================================")


###############################################################################
# main function: train + auto resume + threshold tuning + batch inference
###############################################################################
def main():
    args = parse_args()

    # for reproducibility
    set_seed(args.seed)

    # load dataset
    dataset_dict = load_dataset_dict(args.train_csv, args.valid_csv)

    # tokenizer
    model_name = "xlm-roberta-large"
    logging.info(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # tokenize
    def token_map_fn(batch):
        return tokenize_batch(batch, tokenizer, args.max_length)

    logging.info("Tokenizing datasets ...")
    dataset_dict = dataset_dict.map(token_map_fn, batched=True)
    # remove original text column
    dataset_dict = dataset_dict.remove_columns(["text"])

    # model setup
    logging.info("Initializing XLM-RoBERTa-large for multi-label classification ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=7, problem_type="multi_label_classification"
    )
    # if gradient checkpointing
    if args.gradient_checkpointing:
        logging.info("Enabling gradient checkpointing to reduce VRAM usage.")
        model.gradient_checkpointing_enable()

    # training args
    logging.info("Configuring TrainingArguments ...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.hub_model_id else None,
        deepspeed=args.deepspeed,
    )

    # custom trainer
    trainer = CustomWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        # pass partial to compute_metrics so we can easily handle thresholds
        compute_metrics=lambda eval_pred: compute_multilabel_metrics_per_label(
            eval_pred.predictions,
            eval_pred.label_ids,
            categories=[
                "race",
                "religion",
                "gender",
                "sexual_orientation",
                "nationality",
                "disability",
                "misc_hate",
            ],
            # by default we won't do label-specific thresholds here (0.5). We'll do a
            # second evaluation after we find thresholds if --tune_thresholds is True.
            label_thresholds=None,
        ),
    )

    # for class weighting
    if args.class_weights:
        logging.info(
            "[ClassWeights] Computing label frequencies for weighted BCE loss ..."
        )
        train_data = dataset_dict["train"]
        freqs, total = compute_label_frequencies(train_data, num_labels=7)
        weights = compute_class_weights(freqs, total)
        logging.info(
            f"Label frequencies = {freqs}, total samples = {total}, computed weights = {weights}"
        )
        trainer.set_class_weights(torch.tensor(weights, dtype=torch.float))

    # early stopping
    if args.early_stopping_patience > 0:
        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )

    # sample based checkpoint callback
    sample_checkpoint_cb = SampleCheckpointCallback(
        checkpoint_interval=args.sample_checkpoint_interval, output_dir=args.output_dir
    )
    trainer.add_callback(sample_checkpoint_cb)

    # auto resume
    resume_checkpoint = None
    if args.auto_resume_from_latest:
        resume_checkpoint = find_latest_sample_checkpoint(args.output_dir)

    # train
    logging.info(
        "Beginning training with XLM-RoBERTa-large on 7-label hate speech data ..."
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # evaluate (with default thresholds=0.5)
    logging.info("Evaluating on validation set with threshold=0.5 ...")
    base_metrics = trainer.evaluate()
    logging.info(f"Validation Metrics (no threshold tuning): {base_metrics}")

    # save model
    logging.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # threshold tuning
    # if user wants to find label-specific thresholds on the validation set
    final_label_thresholds = [0.5] * 7
    categories = [
        "race",
        "religion",
        "gender",
        "sexual_orientation",
        "nationality",
        "disability",
        "misc_hate",
    ]
    if args.tune_thresholds:
        logging.info(
            "[ThresholdTuning] Generating predictions on validation set for threshold search ..."
        )
        # manually get the validation set probs, do a normal predictions call:
        val_preds = trainer.predict(trainer.eval_dataset)
        # val_preds.predictions shape => [num_samples, 7], these are logits
        # val_preds.label_ids shape  => [num_samples, 7]
        val_logits = val_preds.predictions  # shape (N,7)
        val_labels = val_preds.label_ids  # shape (N,7)
        # convert to probabilities
        sigmoid = nn.Sigmoid()
        val_probs = sigmoid(torch.tensor(val_logits)).numpy()
        # now find best threshold for each label
        final_label_thresholds = find_optimal_thresholds(
            val_probs, val_labels, categories
        )

        # evaluate with these thresholds
        tuned_metrics = evaluate_with_thresholds(
            val_probs, val_labels, final_label_thresholds, categories
        )
        logging.info(
            f"[ThresholdTuning] Metrics with tuned thresholds: {tuned_metrics}"
        )
    else:
        logging.info("No threshold tuning requested (--tune_thresholds False).")

    # if push to hub was set
    if args.push_to_hub:
        trainer.push_to_hub()
        logging.info("Pushed model to Hugging Face Hub.")

    # inference demo
    if args.run_inference:
        logging.info(
            "run_inference=True: We'll do a demonstration of batch inference using final thresholds ..."
        )
        final_model_path = args.output_dir

        example_texts = [
            "I hate all people. Everybody is a criminal.",
            "Hi, how are you doing?",
            "Go back to your country, we don't want you here.",
        ]
        print("===================================================================")
        print(
            "Multi-label inference demonstration. We'll classify a sample text with label-specific thresholds (if any)."
        )
        custom_input = input(
            "Enter a custom text to classify (or press ENTER to skip): "
        ).strip()
        if custom_input:
            example_texts = [custom_input]

        # do batch inference:
        run_demo_inference_single_texts(
            model_path=final_model_path,
            texts=example_texts,
            max_length=args.max_length,
            thresholds=final_label_thresholds,
        )
    else:
        logging.info("Skipping inference (run_inference=False).")


if __name__ == "__main__":
    main()
