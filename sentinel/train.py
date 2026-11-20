"""Train SENTINEL (LoRA or QLoRA) on a local GPU or a Hyperstack VM.

    python -m sentinel.train                       # LoRA, defaults
    python -m sentinel.train --method qlora        # 4-bit QLoRA
    python -m sentinel.train --max-steps 50 --max-eval-rows 2000   # quick check

Produces output/<run-name>/:
  adapter/                LoRA weights + classification head + tokenizer
  sentinel_config.json    base model, method, max_length, reason codes, train settings
  calibration.json        Platt params + REVIEW / BLOCK thresholds (fit on validation)
  metrics_validation.json, metrics_holdout.json
  train_log.json          loss curve
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from sentinel import NUM_LABELS, REASON_CODES
from sentinel.data import balance, load_split, reason_vector, stratified_sample

logger = logging.getLogger("sentinel.train")


class SFTRows(Dataset):
    def __init__(self, rows: List[dict], tokenizer, max_length: int):
        self.rows, self.tok, self.max_length = rows, tokenizer, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["prompt"], truncation=True, max_length=self.max_length)
        target = [float(r["fraud_label"])] + reason_vector(r["reasons"])
        return {"input_ids": enc["input_ids"], "targets": target}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, : len(b["input_ids"])] = torch.tensor(b["input_ids"])
            mask[i, : len(b["input_ids"])] = 1
        return {"input_ids": ids, "attention_mask": mask,
                "targets": torch.tensor([b["targets"] for b in batch], dtype=torch.float32)}


def make_trainer_cls():
    from transformers import Trainer

    class SentinelTrainer(Trainer):
        """BCE on the fraud logit (with pos_weight) + weighted BCE on reason logits."""

        def __init__(self, *a, pos_weight: float = 1.0, reason_weight: float = 0.5, **kw):
            super().__init__(*a, **kw)
            self.pos_weight, self.reason_weight = pos_weight, reason_weight

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            targets = inputs.pop("targets")
            logits = model(**inputs).logits.float()
            bce = torch.nn.functional.binary_cross_entropy_with_logits
            fraud = bce(logits[:, 0], targets[:, 0],
                        pos_weight=torch.tensor(self.pos_weight, device=logits.device))
            reasons = bce(logits[:, 1:], targets[:, 1:])
            loss = fraud + self.reason_weight * reasons
            return (loss, {"logits": logits}) if return_outputs else loss

    return SentinelTrainer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune SENTINEL with LoRA / QLoRA")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--method", choices=["lora", "qlora"], default="lora")
    p.add_argument("--data-dir", default="data/sft")
    p.add_argument("--output-root", default="output")
    p.add_argument("--run-name", default=None)
    p.add_argument("--tag", default="local", help="Label for the run name, e.g. local / hyperstack")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--genuine-per-fraud", type=int, default=10,
                   help="Downsample genuine rows to N per fraud row (0 = keep all)")
    p.add_argument("--pos-weight", type=float, default=1.0, help="BCE pos_weight for the fraud class")
    p.add_argument("--reason-weight", type=float, default=0.5)
    p.add_argument("--eligible-only", action="store_true",
                   help="Train only on rows with eligible_at_next_split_start=true (strict backtest)")
    p.add_argument("--max-train-rows", type=int, default=0, help="Cap training rows after balancing")
    p.add_argument("--max-eval-rows", type=int, default=20000,
                   help="Stratified cap for validation/holdout scoring (0 = all)")
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--skip-holdout", action="store_true")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                   help="Recompute activations: ~6x less GPU memory, ~30%% slower (default on)")
    p.add_argument("--target-recall", type=float, default=0.85)
    p.add_argument("--target-precision", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def run(a) -> Path:
    from transformers import TrainingArguments, set_seed

    from sentinel.evaluate import (apply_platt, calibrate_and_pick_thresholds,
                                   classification_metrics, reason_metrics, score_rows)
    from sentinel.model import build_model

    set_seed(a.seed)
    name = a.run_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{a.method}-{a.tag}"
    run_dir = Path(a.output_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info("Run %s | device=%s | method=%s | base=%s", name, dev, a.method, a.base_model)

    # ---- data
    train_rows = load_split("train", Path(a.data_dir))
    if a.eligible_only:
        train_rows = [r for r in train_rows if r["eligible"]]
    train_rows = balance(train_rows, a.genuine_per_fraud, a.seed)
    if a.max_train_rows:
        train_rows = train_rows[: a.max_train_rows]
    logger.info("Training rows: %d (%d fraud)", len(train_rows), sum(r["fraud_label"] for r in train_rows))
    val_rows = stratified_sample(load_split("validation", Path(a.data_dir)), a.max_eval_rows, a.seed)

    # ---- model
    model, tok = build_model(a.base_model, a.method, a.lora_r, a.lora_alpha, a.lora_dropout,
                             a.gradient_checkpointing)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=a.warmup_ratio,
        weight_decay=0.0,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        optim="paged_adamw_8bit" if a.method == "qlora" else "adamw_torch",
        remove_unused_columns=False,
        label_names=["targets"],
        dataloader_num_workers=0,
        seed=a.seed,
    )
    trainer = make_trainer_cls()(
        model=model, args=args,
        train_dataset=SFTRows(train_rows, tok, a.max_length),
        data_collator=Collator(tok.pad_token_id),
        pos_weight=a.pos_weight, reason_weight=a.reason_weight,
    )
    t0 = time.time()
    result = trainer.train()
    train_s = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    logger.info("Training done in %.1fs, loss=%.4f, peak GPU memory %.1f GiB",
                train_s, result.training_loss, peak_gb)
    log_history = trainer.state.log_history

    # Release optimizer state, gradients and cached blocks before scoring; otherwise a 24 GB card
    # stays full and evaluation crawls (Windows pages VRAM to system RAM).
    import gc

    model.zero_grad(set_to_none=True)
    trainer.accelerator.free_memory()
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- save
    model.save_pretrained(run_dir / "adapter")
    tok.save_pretrained(run_dir / "adapter")
    meta = {
        "base_model": a.base_model, "method": a.method, "max_length": a.max_length,
        "num_labels": NUM_LABELS, "reason_codes": REASON_CODES, "prompt_format": "fmt-v1",
        "device": dev, "host": socket.gethostname(), "train_runtime_s": round(train_s, 1),
        "train_loss": result.training_loss, "train_rows": len(train_rows),
        "args": vars(a),
    }
    (run_dir / "sentinel_config.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "train_log.json").write_text(json.dumps(log_history, indent=2))

    # ---- validation: calibrate + thresholds
    model.eval()
    logits, labels, reasons = score_rows(model, tok, val_rows, a.max_length, a.eval_batch_size)
    cal = calibrate_and_pick_thresholds(logits[:, 0], labels, a.target_recall, a.target_precision)
    (run_dir / "calibration.json").write_text(json.dumps(cal, indent=2))
    probs = apply_platt(logits[:, 0], cal)
    val = {"split": "validation",
           "calibrated": classification_metrics(probs, labels, cal["review_threshold"], cal["block_threshold"]),
           "reasons": reason_metrics(logits[:, 1:], reasons)}
    (run_dir / "metrics_validation.json").write_text(json.dumps(val, indent=2))
    logger.info("Validation: %s", json.dumps({k: v for k, v in val["calibrated"].items() if k != "curves"}))

    # ---- holdout (acceptance split, thresholds frozen from validation)
    if not a.skip_holdout:
        ho_rows = stratified_sample(load_split("holdout", Path(a.data_dir)), a.max_eval_rows, a.seed)
        logits, labels, reasons = score_rows(model, tok, ho_rows, a.max_length, a.eval_batch_size)
        probs = apply_platt(logits[:, 0], cal)
        ho = {"split": "holdout",
              "calibrated": classification_metrics(probs, labels, cal["review_threshold"], cal["block_threshold"]),
              "reasons": reason_metrics(logits[:, 1:], reasons)}
        (run_dir / "metrics_holdout.json").write_text(json.dumps(ho, indent=2))
        logger.info("Holdout: %s", json.dumps({k: v for k, v in ho["calibrated"].items() if k != "curves"}))

    logger.info("Artifacts in %s", run_dir)
    print(str(run_dir))
    return run_dir


def main(argv: Optional[list] = None):
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
