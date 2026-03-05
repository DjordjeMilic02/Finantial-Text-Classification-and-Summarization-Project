import os
import re
import csv
import json
import time
import platform
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    __version__ as transformers_version,
)

import torch

MODEL_ID = "facebook/bart-large-cnn"
DATASET_ID = "kdave/Indian_Financial_News"
QUICK_TEST = True

rouge = evaluate.load("rouge")


def preprocess_data(examples, tokenizer, max_input_length=1024, max_target_length=128):
    inputs = examples["Content"]
    targets = examples["Summary"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )["input_ids"]

    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]
    model_inputs["labels"] = labels

    return model_inputs


def main():
    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_raw = dataset["train"]
    eval_raw = dataset["test"]

    if QUICK_TEST:
        train_raw = train_raw.shuffle(seed=42).select(range(len(train_raw) // 10))
        eval_raw = eval_raw.shuffle(seed=42).select(range(len(eval_raw) // 10))
        num_train_epochs = 2
        print(f"[INFO] Quick test: {len(train_raw)} train / {len(eval_raw)} eval samples")
    else:
        num_train_epochs = 3
        print(f"[INFO] Full run: {len(train_raw)} train / {len(eval_raw)} eval samples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    train_dataset = train_raw.map(
        lambda batch: preprocess_data(batch, tokenizer),
        batched=True,
        remove_columns=train_raw.column_names,
    )
    eval_dataset = eval_raw.map(
        lambda batch: preprocess_data(batch, tokenizer),
        batched=True,
        remove_columns=eval_raw.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = [
            [token if token != -100 else tokenizer.pad_token_id for token in label]
            for label in labels
        ]
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=preds, references=labels, use_stemmer=True)
        return {key: value * 100 for key, value in result.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir="./bart-financial-finetuned",
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=num_train_epochs,
        save_total_limit=2,
        generation_max_length=128,
        generation_num_beams=4,
        predict_with_generate=True,
        fp16=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    total_time_s = time.time() - t0
    print(f"[INFO] Training time: {total_time_s:.1f}s")

    print("[INFO] Evaluating on eval set…")
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print(eval_metrics)

    FINAL_DIR = Path("./bart-financial-finetuned-final")
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    with (FINAL_DIR / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in eval_metrics.items()}, f, indent=2)

    history = trainer.state.log_history
    per_epoch: dict = {}
    for row in history:
        if "eval_loss" in row and "epoch" in row:
            ep = int(round(row["epoch"]))
            per_epoch[ep] = {
                "epoch": ep,
                "eval_loss": row.get("eval_loss"),
                "rouge1": row.get("eval_rouge1"),
                "rouge2": row.get("eval_rouge2"),
                "rougeL": row.get("eval_rougeL"),
                "rougeLsum": row.get("eval_rougeLsum"),
            }

    if per_epoch:
        rows = [per_epoch[e] for e in sorted(per_epoch.keys())]
        with (FINAL_DIR / "metrics_per_epoch.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["epoch", "eval_loss", "rouge1", "rouge2", "rougeL", "rougeLsum"]
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)

        epochs = [r["epoch"] for r in rows]
        plt.figure(figsize=(7, 4.5))
        for key, label in [
            ("rouge1", "ROUGE-1 (F1)"),
            ("rouge2", "ROUGE-2 (F1)"),
            ("rougeL", "ROUGE-L (F1)"),
            ("rougeLsum", "ROUGE-Lsum (F1)"),
        ]:
            vals = [r[key] for r in rows]
            if any(v is not None for v in vals):
                plt.plot(epochs, vals, marker="o", label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Evaluation Metrics per Epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FINAL_DIR / "metrics_per_epoch.png", dpi=150)
        plt.close()

    try:
        n_samples = min(64, len(eval_raw))
        sample_raw = eval_raw.select(range(n_samples))
        pred_output = trainer.predict(
            eval_dataset.select(range(n_samples))
        )
        pred_ids = pred_output.predictions
        preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        refs = sample_raw["Summary"]
        inputs = sample_raw["Content"]

        with (FINAL_DIR / "predictions_head.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "input", "reference", "prediction"])
            for i, (inp, ref, pr) in enumerate(zip(inputs, refs, preds)):
                w.writerow([i, inp, ref, pr])
    except Exception as e:
        print("[WARN] Could not save sample predictions:", e)

    device = "cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    gpu_name = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    run_settings = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "quick_test": QUICK_TEST,
        "dataset_sizes": {"train": len(train_raw), "eval": len(eval_raw)},
        "training_args": training_args.to_dict(),
        "device": device,
        "gpu_name": gpu_name,
        "total_training_time_seconds": round(float(total_time_s), 3),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "python_version": platform.python_version(),
    }
    with (FINAL_DIR / "run_settings.json").open("w", encoding="utf-8") as f:
        json.dump(run_settings, f, indent=2)

    trainer.save_model("./bart-financial-finetuned-final")
    tokenizer.save_pretrained("./bart-financial-finetuned-final")

    print("Fine-tuning complete. Model and artifacts saved in ./bart-financial-finetuned-final")


if __name__ == "__main__":
    main()
