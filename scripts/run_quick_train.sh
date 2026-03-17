#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

source .venv/bin/activate

python -m src.train \
  --data-dir data/packed \
  --checkpoints-dir artifacts/checkpoints \
  --vocab-size 1024 --seq-len 256 --d-model 128 --n-layers 7 --n-heads 4 \
  --batch-tokens 8192 --steps 2000 \
  --lr 3e-4 --warmup-steps 200 --scheduler cosine --weight-decay 0.1 --grad-clip 1.0 \
  --eval-interval 200 --save-interval 200 \
  --sp-model artifacts/tokenizer/sp_bpe_1k.model \
  --sample-prompt "Once upon a time," --sample-max-new-tokens 120


