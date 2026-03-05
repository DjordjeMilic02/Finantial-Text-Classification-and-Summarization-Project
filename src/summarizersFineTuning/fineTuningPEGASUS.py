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
import numpy as np
import os
import argparse
import torch
import time
import json
import csv
import platform
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="google/pegasus-large")
    p.add_argument("--dataset_id", type=str, default="soumakchak/earnings_call_dataset")
    p.add_argument("--output_dir", type=str, default="./pegasus-earnings-fast")
    p.add_argument("--max_input_length", type=int, default=768)
    p.add_argument("--max_target_length", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--quick_test", action="store_true")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2))
    return p.parse_args()

def preprocess_batch(examples, tokenizer, max_input_length=768, max_target_length=96):
    inputs = examples["document"]
    targets = examples["summary"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")["input_ids"]
    labels = [[(tok if tok != tokenizer.pad_token_id else -100) for tok in seq] for seq in labels]
    model_inputs["labels"] = labels
    return model_inputs

def build_compute_metrics(tokenizer, rouge):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v * 100, 2) for k, v in scores.items()}
    return compute_metrics

def maybe_freeze_encoder(model, freeze):
    if not freeze:
        return
    for p in model.get_encoder().parameters():
        p.requires_grad = False

def save_history_plots_and_csv(trainer, out_dir):
    history = trainer.state.log_history
    rows = []
    for row in history:
        if "epoch" in row and any(k.startswith("eval_") for k in row.keys()):
            rows.append({
                "epoch": int(round(row["epoch"])),
                "eval_loss": row.get("eval_loss"),
                "rouge1": row.get("eval_rouge1"),
                "rouge2": row.get("eval_rouge2"),
                "rougeL": row.get("eval_rougeL"),
                "rougeLsum": row.get("eval_rougeLsum"),
            })
    if not rows:
        return
    rows_by_epoch = {}
    for r in rows:
        rows_by_epoch[r["epoch"]] = r
    rows = [rows_by_epoch[e] for e in sorted(rows_by_epoch.keys())]
    with (out_dir / "metrics_per_epoch.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "eval_loss", "rouge1", "rouge2", "rougeL", "rougeLsum"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    epochs = [r["epoch"] for r in rows]
    plt.figure(figsize=(7, 4.5))
    for key, label in [("rouge1", "ROUGE-1 (F1)"), ("rouge2", "ROUGE-2 (F1)"), ("rougeL", "ROUGE-L (F1)"), ("rougeLsum", "ROUGE-Lsum (F1)")]:
        vals = [r[key] for r in rows]
        if any(v is not None for v in vals):
            plt.plot(epochs, vals, marker="o", label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics per Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_per_epoch.png", dpi=150)
    plt.close()

def main():
    args = build_args()
    os.environ.setdefault("WANDB_DISABLED", "true")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        use_safetensors=True,
        device_map="auto",
        trust_remote_code=False,
        local_files_only=False,
        dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    maybe_freeze_encoder(model, args.freeze_encoder)
    ds = load_dataset(args.dataset_id)
    if args.quick_test:
        ds["train"] = ds["train"].select(range(min(200, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(50, len(ds["validation"]))))
        ds["test"] = ds["test"].select(range(min(50, len(ds["test"]))))
    preprocess = lambda batch: preprocess_batch(batch, tokenizer, args.max_input_length, args.max_target_length)
    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names, num_proc=args.num_proc)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, pad_to_multiple_of=8)
    rouge = evaluate.load("rouge")
    compute_metrics = build_compute_metrics(tokenizer, rouge)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.beam_size,
        fp16=args.fp16,
        save_total_limit=2,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="rougeLsum",
        greater_is_better=True,
        optim="adafactor",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        torch_compile=False,
        save_safetensors=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    t0 = time.time()
    train_metrics = None
    err_text = None
    try:
        train_output = trainer.train()
        train_metrics = train_output.metrics
    except Exception as e:
        err_text = str(e)
    total_time_s = time.time() - t0
    trainer.save_state()
    if train_metrics is not None:
        trainer.save_metrics("train", train_metrics)
    save_history_plots_and_csv(trainer, out_dir)
    eval_metrics = trainer.evaluate(tokenized["validation"])
    trainer.save_metrics("eval", eval_metrics)
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    trainer.save_metrics("test", test_metrics)
    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({k: float(v) if isinstance(v, (int, float)) else (float(v.item()) if hasattr(v, "item") else str(v)) for k, v in test_metrics.items()}, f, indent=2)
    if err_text is not None:
        with (out_dir / "train_error.txt").open("w", encoding="utf-8") as f:
            f.write(err_text)
    if len(ds["test"]) > 0:
        raw_doc = ds["test"][0]["document"]
        inputs = tokenizer([raw_doc], max_length=args.max_input_length, truncation=True, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        summary_ids = model.generate(**inputs, max_length=args.max_target_length, num_beams=args.beam_size, length_penalty=0.9, early_stopping=True)
        with (out_dir / "sample_summary.txt").open("w", encoding="utf-8") as f:
            f.write(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    try:
        n_samples = min(64, len(ds["test"]))
        if n_samples > 0:
            preds_out = trainer.predict(tokenized["test"].select(range(n_samples)))
            pred_ids = preds_out.predictions
            preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            subset_raw = ds["test"].select(range(n_samples))
            refs = subset_raw["summary"]
            inputs_raw = subset_raw["document"]
            with (out_dir / "predictions_head.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["idx", "input", "reference", "prediction"])
                for i, (inp, ref, pr) in enumerate(zip(inputs_raw, refs, preds)):
                    w.writerow([i, inp, ref, pr])
    except Exception as e:
        with (out_dir / "predictions_error.txt").open("w", encoding="utf-8") as f:
            f.write(str(e))
    device = "cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    run_settings = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "quick_test": args.quick_test,
        "dataset_sizes": {"train": len(ds["train"]), "validation": len(ds["validation"]), "test": len(ds["test"])},
        "training_args": training_args.to_dict(),
        "device": device,
        "gpu_name": gpu_name,
        "total_training_time_seconds": round(float(total_time_s), 3),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "python_version": platform.python_version(),
    }
    with (out_dir / "run_settings.json").open("w", encoding="utf-8") as f:
        json.dump(run_settings, f, indent=2)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(str(out_dir.resolve()))

if __name__ == "__main__":
    main()
