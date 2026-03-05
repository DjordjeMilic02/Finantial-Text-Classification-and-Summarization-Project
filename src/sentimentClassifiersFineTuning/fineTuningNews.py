import os
import csv
from collections import Counter
from typing import Any, List, Dict
import numpy as np

import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

from datasets import Dataset, DatasetDict
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
)
from transformers.trainer import Trainer as HFTrainer

from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

CSV_PATH = "./data/opendatabay_financial_news_400.csv"

MODEL_ID = "allenai/longformer-base-4096"
OUTPUT_DIR = "./fullnews-longformer-opendatabay"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_RATIO = 0.10
VAL_RATIO  = 0.10

MAX_LENGTH = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 1e-5
NUM_EPOCHS = 10
USE_FP16 = True

USE_WEIGHTED_LOSS = True
USE_FOCAL = False
USE_WEIGHTED_SAMPLER = False
FOCAL_GAMMA = 1.75

LABEL_ORDER = ["positive", "negative"]

NEG_UPWEIGHT = 1.6

def canon_label(x: Any) -> str:
    if isinstance(x, str):
        z = x.strip().lower()
        if z in ["pos","positive","+"]: return "positive"
        if z in ["neg","negative","-"]: return "negative"
    if isinstance(x, (int, np.integer)):
        return "positive" if int(x) == 1 else "negative"
    return ""

def make_label_maps(labels: List[str]):
    uniq = sorted(set(labels), key=lambda s: LABEL_ORDER.index(s) if s in LABEL_ORDER else 99)
    label2id = {lbl: i for i, lbl in enumerate(uniq)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label

def build_metrics():
    acc = evaluate.load("accuracy")
    f1  = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            **acc.compute(predictions=preds, references=labels),
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }
    return compute_metrics

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=FOCAL_GAMMA, reduction="mean"):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()

class WeightedTrainer(HFTrainer):
    def __init__(self, class_weights=None, focal=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
        self.loss_fct = FocalLoss(alpha=alpha) if focal else nn.CrossEntropyLoss(weight=alpha)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def make_weighted_sampler(dataset, num_labels, neg_id):
    labels = [int(ex["labels"]) for ex in dataset]
    counts = np.bincount(labels, minlength=num_labels).astype(float)
    inv = 1.0 / np.clip(counts, 1, None)
    weights = [inv[y] for y in labels]
    for i, y in enumerate(labels):
        if y == neg_id:
            weights[i] *= NEG_UPWEIGHT
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

def read_opendatabay_csv(path: str) -> Dataset:
    assert os.path.exists(path), f"CSV not found at {path}. Please place the file there."
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            headline = (r.get("Headline") or "").strip()
            synopsis = (r.get("Synopsis") or "").strip()
            fulltxt  = (r.get("Full_text") or "").strip()
            status   = (r.get("Final Status") or r.get("Final_Status") or r.get("Label") or "").strip()
            parts = [p for p in [headline, synopsis, fulltxt] if p]
            text = " [SEP] ".join(parts)
            y = canon_label(status)
            if text and y in {"positive","negative"}:
                rows.append({"text": text, "label": y})
    if not rows:
        raise RuntimeError("No valid rows parsed from CSV (check column names and encoding).")
    return Dataset.from_dict({
        "text": [r["text"] for r in rows],
        "label": [r["label"] for r in rows],
    })

def save_matrix_csv(path: str, matrix: np.ndarray, id2label: Dict[int, str], fmt: str | None = None):
    num_labels = matrix.shape[0]
    labels = [id2label[i] for i in range(num_labels)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for i in range(num_labels):
            row = matrix[i]
            if fmt is not None:
                writer.writerow([labels[i]] + [format(x, fmt) for x in row])
            else:
                writer.writerow([labels[i]] + list(row))

def main():
    os.environ.setdefault("TRANSFORMERS_USE_SAFE_TENSORS", "1")

    full_ds = read_opendatabay_csv(CSV_PATH)
    try:
        s1 = full_ds.train_test_split(test_size=TEST_RATIO, seed=SEED, stratify_by_column="label")
    except Exception:
        s1 = full_ds.train_test_split(test_size=TEST_RATIO, seed=SEED)
    pool, test_ds = s1["train"], s1["test"]
    try:
        s2 = pool.train_test_split(test_size=VAL_RATIO, seed=SEED, stratify_by_column="label")
    except Exception:
        s2 = pool.train_test_split(test_size=VAL_RATIO, seed=SEED)
    train_ds, val_ds = s2["train"], s2["test"]
    ds = DatasetDict(train=train_ds, validation=val_ds, test=test_ds)

    print("Sizes ->", {k: len(v) for k, v in ds.items()})
    print("Label counts (train):", Counter([ex["label"] for ex in ds["train"]]))
    print("Label counts (val):  ", Counter([ex["label"] for ex in ds["validation"]]))
    print("Label counts (test): ", Counter([ex["label"] for ex in ds["test"]]))

    label2id, id2label = make_label_maps([ex["label"] for ex in ds["train"]])
    print("label2id:", label2id, "id2label:", id2label)
    pos_id = label2id["positive"]
    neg_id = label2id["negative"]

    def to_ids(example):
        example["labels"] = label2id.get(example["label"], -1)
        return example

    ds = DatasetDict({k: v.map(to_ids) for k, v in ds.items()})
    ds = DatasetDict({k: v.filter(lambda ex: ex["labels"] >= 0) for k, v in ds.items()})

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False)
    tokenized = DatasetDict({
        k: v.map(
            tokenize,
            batched=True,
            remove_columns=[c for c in v.column_names if c not in ["labels"]],
        )
        for k, v in ds.items()
    })

    num_labels = len(label2id)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True,
    )

    class_weights = None
    if USE_WEIGHTED_LOSS:
        counts = np.bincount([int(ex["labels"]) for ex in tokenized["train"]], minlength=num_labels).astype(float)
        inv = 1.0 / np.clip(counts, 1, None)
        class_weights = (inv / inv.sum()) * num_labels
        class_weights[neg_id] *= NEG_UPWEIGHT
        print("Class counts:", counts, "=> class_weights:", class_weights)

    train_sampler = None
    if USE_WEIGHTED_SAMPLER:
        train_sampler = make_weighted_sampler(tokenized["train"], num_labels=num_labels, neg_id=neg_id)

    data_collator = DataCollatorWithPadding(tokenizer=tok)
    metrics_fn = build_metrics()
    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        seed=SEED,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        fp16=USE_FP16,
        report_to=[],
        remove_unused_columns=False,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights if USE_WEIGHTED_LOSS else None,
        focal=USE_FOCAL,
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
    )

    if train_sampler is not None:
        def _get_train_dataloader(self):
            from torch.utils.data import DataLoader
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        trainer.get_train_dataloader = _get_train_dataloader.__get__(trainer, type(trainer))

    trainer.train()

    history = trainer.state.log_history
    per_epoch = {}
    for row in history:
        if "eval_loss" in row and "epoch" in row:
            ep = int(round(row["epoch"]))
            per_epoch[ep] = {
                "epoch": ep,
                "eval_loss": row.get("eval_loss"),
                "eval_accuracy": row.get("eval_accuracy"),
                "eval_f1_macro": row.get("eval_f1_macro"),
            }
    if per_epoch:
        rows = [per_epoch[e] for e in sorted(per_epoch.keys())]
        csv_path = os.path.join(OUTPUT_DIR, "metrics_per_epoch.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "eval_loss", "eval_accuracy", "eval_f1_macro"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        epochs = [r["epoch"] for r in rows]
        accs = [r["eval_accuracy"] for r in rows]
        f1s  = [r["eval_f1_macro"] for r in rows]
        plt.figure(figsize=(7, 4.5))
        if any(v is not None for v in accs):
            plt.plot(epochs, accs, marker="o", label="Accuracy")
        if any(v is not None for v in f1s):
            plt.plot(epochs, f1s, marker="o", label="Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Eval Metrics per Epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "metrics_per_epoch.png"), dpi=150)
        plt.close()

    print("\n=== Test metrics (argmax) ===")
    test_out = trainer.predict(tokenized["test"])
    test_logits = test_out.predictions
    test_labels = test_out.label_ids
    test_preds  = test_logits.argmax(axis=1)

    test_acc = accuracy_score(test_labels, test_preds)
    print(f"test_accuracy: {test_acc:.4f}")
    for k, v in test_out.metrics.items():
        if k.startswith("test_"):
            print(f"{k}: {v:.4f}")

    num_labels = len(id2label)
    cm = confusion_matrix(test_labels, test_preds, labels=[pos_id, neg_id])
    print("\n=== Confusion Matrix (argmax; rows=true, cols=pred) ===")
    header = ["{:>10}".format("")] + ["{:>10}".format(id2label[i]) for i in [pos_id, neg_id]]
    print(" ".join(header))
    for i in [pos_id, neg_id]:
        row = ["{:>10}".format(id2label[i])] + ["{:>10}".format(cm[[pos_id,neg_id].index(i), j]) for j in range(cm.shape[1])]
        print(" ".join(row))

    acc_from_cm = (cm.trace() / cm.sum()) if cm.sum() > 0 else 0.0
    print(f"\nAccuracy from confusion matrix: {acc_from_cm:.4f}")
    if abs(acc_from_cm - test_acc) > 1e-6:
        print("WARNING: accuracy mismatch between metric and CM!")

    raw_cm_csv = os.path.join(OUTPUT_DIR, "confusion_matrix_argmax.csv")
    save_matrix_csv(raw_cm_csv, cm.astype(int), {i:id2label[i] for i in [pos_id,neg_id]})
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums
    norm_cm_csv = os.path.join(OUTPUT_DIR, "confusion_matrix_argmax_normalized.csv")
    save_matrix_csv(norm_cm_csv, cm_norm.astype(float), {i:id2label[i] for i in [pos_id,neg_id]}, fmt=".6f")

    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
