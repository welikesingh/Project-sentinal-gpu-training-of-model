"""SENTINEL: LoRA/QLoRA fine-tuning of Qwen2.5 for fraud scoring.

Modules
  data      - download + load the SFT splits from the Supabase bucket
  model     - base model + LoRA/QLoRA sequence-classification heads
  train     - training entry point (local GPU or Hyperstack VM)
  evaluate  - metrics, Platt calibration, threshold selection
  infer     - scorer, CLI and a tiny HTTP server returning JSON decisions
"""

REASON_CODES = [
    "R01_NEW_DEVICE",
    "R02_GEO_MISMATCH",
    "R03_ODD_HOUR",
    "R04_FAILED_ATTEMPTS",
    "R05_AMOUNT_SPIKE",
    "R06_NEW_ACCOUNT",
    "R07_VELOCITY",
    "R08_HIGH_RISK_MCC",
]

# Output head layout: index 0 = fraud logit, 1..8 = reason-code logits.
NUM_LABELS = 1 + len(REASON_CODES)
