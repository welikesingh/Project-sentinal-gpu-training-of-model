"""Model construction and batched scoring.

The base causal LM (Qwen2.5-0.5B-Instruct by default) is loaded as a
sequence-classification model with NUM_LABELS outputs:
  logit[0]   -> fraud probability (sigmoid)
  logit[1:]  -> one logit per reason code (multi-label)
LoRA adapters go on all linear layers; the `score` head is trained fully
(modules_to_save) and saved inside the adapter directory.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sentinel import NUM_LABELS

logger = logging.getLogger(__name__)


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base(name_or_path: str, method: str, pad_token_id: int):
    """Load the base model with a fresh NUM_LABELS-way head. method: lora | qlora."""
    from transformers import AutoModelForSequenceClassification

    kwargs: dict = {"num_labels": NUM_LABELS, "dtype": pick_dtype()}
    if method == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA (4-bit) needs a CUDA GPU; use --method lora on CPU")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=pick_dtype(),
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForSequenceClassification.from_pretrained(name_or_path, **kwargs)
    model.config.pad_token_id = pad_token_id
    if method != "qlora" and torch.cuda.is_available():
        model.to("cuda")
    return model


def build_model(base_model: str, method: str = "lora", r: int = 16, alpha: int = 32,
                dropout: float = 0.05, gradient_checkpointing: bool = False):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    tok = load_tokenizer(base_model)
    model = load_base(base_model, method, tok.pad_token_id)
    if method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",
        modules_to_save=["score"],
        bias="none",
    )
    model = get_peft_model(model, cfg)
    # Keep the trainable head in fp32 for stable optimisation.
    for _, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return model, tok


def load_trained(run_dir: str, device: Optional[str] = None):
    """Load base + trained adapter from a run directory produced by sentinel.train."""
    import json
    from pathlib import Path

    from peft import PeftModel

    run = Path(run_dir)
    meta = json.loads((run / "sentinel_config.json").read_text())
    tok = load_tokenizer(str(run / "adapter"))
    base = load_base(meta["base_model"], meta["method"], tok.pad_token_id)
    model = PeftModel.from_pretrained(base, str(run / "adapter"))
    if meta["method"] != "qlora":
        model = model.merge_and_unload()
        if device:
            model.to(device)
    model.eval()
    return model, tok, meta


@torch.no_grad()
def predict_logits(model, tokenizer, prompts: List[str], max_length: int = 512,
                   batch_size: int = 32) -> torch.Tensor:
    """Return [N, NUM_LABELS] float32 logits (input order). Batches are length-sorted for speed."""
    import contextlib

    model.eval()
    device = next(model.parameters()).device
    # Autocast keeps an fp32 head and a bf16/fp16 body compatible outside the Trainer.
    amp = (torch.autocast("cuda", dtype=pick_dtype()) if device.type == "cuda"
           else contextlib.nullcontext())
    # Longest first: the first batch reserves the biggest blocks and later batches reuse them.
    # Shortest-first fragments the CUDA cache (11.8 GiB reserved vs 1.7 GiB measured on QLoRA).
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    out = torch.empty(len(prompts), NUM_LABELS)
    import time

    n_batches = (len(order) + batch_size - 1) // batch_size
    every = max(1, n_batches // 10) if len(prompts) >= 2000 else 0
    t0 = time.time()
    with amp:
        for b, s in enumerate(range(0, len(order), batch_size), 1):
            idx = order[s:s + batch_size]
            enc = tokenizer([prompts[i] for i in idx], truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(device)
            out[idx] = model(**enc).logits.float().cpu()
            if every and (b % every == 0 or b == n_batches):
                done = min(s + batch_size, len(order))
                logger.info("Scoring %d/%d rows (%.0f rows/s)", done, len(order), done / max(1e-6, time.time() - t0))
    return out
