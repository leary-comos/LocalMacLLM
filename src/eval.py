from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import mlx.core as mx

from src.datasets.tinystories_dataset import create_loader
from src.model.gpt_mlx import GPT, ModelConfig, count_parameters


def cross_entropy_loss(logits: mx.array, targets: mx.array) -> mx.array:
    # logits: (B, T, V), targets: (B, T) int
    # compute log-softmax and gather NLL
    logsumexp = mx.logsumexp(logits, axis=-1, keepdims=True)
    log_probs = logits - logsumexp
    nll = -mx.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return mx.mean(nll)


def evaluate(
    data_dir: Path,
    *,
    split: str,
    cfg: ModelConfig,
    checkpoint: Optional[Path] = None,
    max_batches: Optional[int] = None,
) -> dict:
    model = GPT(cfg)

    # Attempt to load weights if a checkpoint is provided
    if checkpoint is not None and checkpoint.exists():
        try:
            # For now, checkpoints are metadata JSON; model weights remain current
            with checkpoint.open("r", encoding="utf-8") as f:
                _meta = json.load(f)
        except Exception as e:  # pragma: no cover
            print(f"Warning: failed to read checkpoint metadata '{checkpoint}': {e}")

    # Build loader
    loader = create_loader(
        packed_dir=data_dir,
        split=split,
        target_batch_tokens=8192,
        shuffle=False,
        drop_last=False,
    )

    total_tokens = 0
    total_loss = 0.0

    for i, batch_np in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        # Next-token prediction: inputs shift right vs targets
        # batch_np: (B, T)
        if batch_np.shape[1] < 2:
            continue
        inputs_np = batch_np[:, :-1]
        targets_np = batch_np[:, 1:]

        inputs = mx.array(inputs_np)
        targets = mx.array(targets_np)

        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        # Accumulate loss weighted by number of tokens
        tokens_in_batch = targets_np.size
        total_tokens += int(tokens_in_batch)
        total_loss += float(loss.item()) * tokens_in_batch

    if total_tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "tokens": 0}

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return {"loss": avg_loss, "perplexity": ppl, "tokens": total_tokens, "params": count_parameters(model)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation loss/perplexity.")
    parser.add_argument("--data-dir", type=str, required=True, help="Packed data directory (e.g., data/packed)")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate (val/test)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (optional)")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=7)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional limit on number of batches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    result = evaluate(
        data_dir=Path(args.data_dir),
        split=args.split,
        cfg=cfg,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        max_batches=args.max_batches,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


