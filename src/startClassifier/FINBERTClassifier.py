import os, json, time, random, csv, platform
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    __version__ as transformers_version,
)
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

CUSTOM_CB_PATH = "customDataset/parsed_custom_cb.jsonl"
MODEL_ID = "yiyanghkust/finbert-pretrain"
OUTPUT_DIR = Path("./finbert-finetuned")
BATCH_SIZE = 8
MAX_LENGTH = 512
NUM_LABELS = 3
SEED = 42
LABEL2ID = {0: "NEWS", 1: "EARNINGS", 2: "CB_SPEECH"}

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def compute_metrics(pred):
    if isinstance(pred, EvalPrediction):
        logits, labels = pred.predictions, pred.label_ids
    else:
        logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }

def tokenize_examples(examples, tokenizer):
    texts = [str(t) if t is not None else "" for t in examples["text"]]
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    enc["label"] = examples["label"]
    return enc

def keep_only_cols(ds):
    keep = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in ds.column_names:
        keep.append("token_type_ids")
    drop = [c for c in ds.column_names if c not in keep]
    return ds.remove_columns(drop)

def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading datasets...")
    news_ds = load_dataset("ashraq/financial-news-articles", split="train") \
        .shuffle(seed=SEED).select(range(3585)) \
        .map(lambda x: {"text": x["text"], "label": 0})

    earnings1 = load_dataset("yeong-hwan/2024-earnings-call-transcript", split="train")

    def flatten_conversations(example):
        conv = example.get("conversations", None)
        if isinstance(conv, list):
            return {"text": " ".join(turn.get("content", "") for turn in conv), "label": 1}
        return {"text": "", "label": 1}

    earnings1 = earnings1.map(flatten_conversations)

    earnings2 = load_dataset("soumakchak/earnings_call_dataset", split="train") \
        .map(lambda x: {"text": x.get("document", ""), "label": 1})

    earnings_ds = concatenate_datasets([earnings1, earnings2]).shuffle(seed=SEED).select(range(3585))

    central_ds = load_dataset("samchain/bis_central_bank_speeches", split="train") \
        .shuffle(seed=SEED).select(range(2249)) \
        .map(lambda x: {"text": x["text"], "label": 2})

    custom_cb = load_dataset("json", data_files=CUSTOM_CB_PATH, split="train") \
        .map(lambda x: {"text": x["text"], "label": 2})

    dataset = concatenate_datasets([news_ds, earnings_ds, central_ds, custom_cb]).shuffle(seed=SEED)
    print(f"[INFO] Total examples: {len(dataset)}")

    dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_ds, test_ds = dataset["train"], dataset["test"]

    def label_counts(ds):
        vals, counts = np.unique(ds["label"], return_counts=True)
        return {LABEL2ID[int(v)]: int(c) for v, c in zip(vals, counts)}

    train_label_counts = label_counts(train_ds)
    test_label_counts  = label_counts(test_ds)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    print("[INFO] Tokenizing dataset...")
    tokenized_train = train_ds.map(lambda ex: tokenize_examples(ex, tokenizer), batched=True, remove_columns=[])
    tokenized_test  = test_ds.map(lambda ex: tokenize_examples(ex, tokenizer), batched=True, remove_columns=[])

    tokenized_train = keep_only_cols(tokenized_train)
    tokenized_test  = keep_only_cols(tokenized_test)

    print("[DEBUG] train columns:", tokenized_train.column_names)
    print("[DEBUG] test  columns:", tokenized_test.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=NUM_LABELS, trust_remote_code=False, use_safetensors=True
    )
    model.config.id2label = {i: LABEL2ID[i] for i in range(NUM_LABELS)}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="steps",
        eval_steps=300,
        logging_steps=100,
        save_strategy="steps",
        save_steps=300,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=16,
        num_train_epochs=12,
        weight_decay=0.02,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        optim="adamw_torch",
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Starting training…")
    t0 = time.time()
    trainer.train()
    total_time_s = time.time() - t0
    print(f"[INFO] Training time: {total_time_s:.1f}s")

    print("[INFO] Evaluating on test set…")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_test)
    print(test_metrics)

    preds_output = trainer.predict(tokenized_test)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    cls_report = classification_report(
        labels, preds, digits=4,
        target_names=[LABEL2ID[i] for i in range(NUM_LABELS)]
    )
    cm = confusion_matrix(labels, preds)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with (OUTPUT_DIR / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    np.savetxt(OUTPUT_DIR / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    with (OUTPUT_DIR / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(cls_report)

    history = trainer.state.log_history
    per_epoch = {}
    for row in history:
        if "eval_loss" in row and "epoch" in row:
            ep = int(round(row["epoch"]))
            per_epoch[ep] = {
                "epoch": ep,
                "eval_loss": row.get("eval_loss"),
                "accuracy": row.get("eval_accuracy"),
                "f1": row.get("eval_f1"),
                "precision": row.get("eval_precision"),
                "recall": row.get("eval_recall"),
            }

    if per_epoch:
        rows = [per_epoch[e] for e in sorted(per_epoch.keys())]
        with (OUTPUT_DIR / "metrics_per_epoch.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch","eval_loss","accuracy","f1","precision","recall"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

        epochs = [r["epoch"] for r in rows]
        plt.figure(figsize=(7, 4.5))
        for key, label in [
            ("accuracy","Accuracy"),
            ("f1","F1 (weighted)"),
            ("precision","Precision (weighted)"),
            ("recall","Recall (weighted)")
        ]:
            vals = [r[key] for r in rows]
            if any(v is not None for v in vals):
                plt.plot(epochs, vals, marker="o", label=label)
        plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Evaluation Metrics per Epoch")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "metrics_per_epoch.png", dpi=150); plt.close()

    device = "cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    gpu_name = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    run_settings = {
        "model_id": MODEL_ID,
        "num_labels": NUM_LABELS,
        "max_length": MAX_LENGTH,
        "training_args": {
            k: (str(v) if not isinstance(v, (int, float, bool, dict, list)) else v)
            for k, v in training_args.to_dict().items()
        },
        "device": device,
        "gpu_name": gpu_name,
        "dataset_sizes": {"train": len(train_ds), "test": len(test_ds)},
        "class_balance": {"train": train_label_counts, "test": test_label_counts},
        "total_training_time_seconds": round(float(total_time_s), 3),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "python_version": platform.python_version(),
    }
    with (OUTPUT_DIR / "run_settings.json").open("w", encoding="utf-8") as f:
        json.dump(run_settings, f, indent=2)

    print("[SUCCESS] Fine-tuning complete! Artifacts in", str(OUTPUT_DIR.resolve()))

if __name__ == "__main__":
    main()
