from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm

from src.datasets.tinystories_dataset import create_loader
from src.model.gpt_mlx import GPT, ModelConfig, count_parameters
import sentencepiece as spm


def cross_entropy_loss(logits: mx.array, targets: mx.array) -> mx.array:
    logsumexp = mx.logsumexp(logits, axis=-1, keepdims=True)
    log_probs = logits - logsumexp
    nll = -mx.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return mx.mean(nll)


def build_scheduler(
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    scheduler: str = "cosine",
):
    def get_lr(step: int) -> float:
        s = max(0, step)
        if s < warmup_steps:
            return base_lr * (s + 1) / max(1, warmup_steps)
        if scheduler == "cosine":
            progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
        return base_lr

    return get_lr


def clip_grads(grads: list[mx.array], max_norm: float) -> list[mx.array]:
    if max_norm <= 0:
        return grads
    # Best-effort: grads structure may be a PyTree; if anything fails, return unmodified
    try:
        def _flatten(node):
            if node is None:
                return []
            if isinstance(node, (list, tuple)):
                out = []
                for n in node:
                    out.extend(_flatten(n))
                return out
            if isinstance(node, dict):
                out = []
                for n in node.values():
                    out.extend(_flatten(n))
                return out
            # Assume mlx array-like leaf
            return [node]

        leaves = _flatten(grads)
        total = 0.0
        for g in leaves:
            try:
                total += float(mx.sum(g * g).item())
            except Exception:
                continue
        global_norm = math.sqrt(total) if total > 0 else 0.0
        if global_norm <= max_norm or global_norm == 0:
            return grads
        scale = max_norm / (global_norm + 1e-8)

        def _scale(node):
            if node is None:
                return None
            if isinstance(node, (list, tuple)):
                return type(node)(_scale(n) for n in node)
            if isinstance(node, dict):
                return {k: _scale(v) for k, v in node.items()}
            try:
                return node * scale
            except Exception:
                return node

        return _scale(grads)
    except Exception:
        return grads


def evaluate_model(model: GPT, val_loader, max_batches: Optional[int] = None) -> dict:
    total_tokens = 0
    total_loss = 0.0
    for i, batch_np in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        if batch_np.shape[1] < 2:
            continue
        inputs = mx.array(batch_np[:, :-1])
        targets = mx.array(batch_np[:, 1:])
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        tokens_in_batch = targets.size
        total_tokens += int(tokens_in_batch)
        total_loss += float(loss.item()) * tokens_in_batch
    if total_tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "tokens": 0}
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return {"loss": avg_loss, "perplexity": ppl, "tokens": total_tokens}


# --- Sampling utils (greedy) for qualitative checks ---
def _sample_greedy(
    model: GPT,
    sp: spm.SentencePieceProcessor,
    special_ids: dict,
    prompt: str,
    seq_len: int,
    max_new_tokens: int,
) -> str:
    # Build initial token list with BOS and prompt
    input_ids = [special_ids['bos']] + sp.encode(prompt, out_type=int)
    # Trim to max context
    input_ids = input_ids[-seq_len:]

    generated: list[int] = []
    for _ in range(max_new_tokens):
        ctx = input_ids[-seq_len:]
        x = mx.array([ctx])
        logits = model(x)
        next_logits = logits[0, -1, :]
        next_token = int(mx.argmax(next_logits).item())
        input_ids.append(next_token)
        if next_token == special_ids['eos']:
            break
        generated.append(next_token)

    # Decode excluding special tokens
    filtered = [t for t in generated if t not in (special_ids['bos'], special_ids['eos'], special_ids['pad'])]
    try:
        return sp.decode(filtered)
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT on TinyStories with MLX")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=7)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from (metadata only for now)")
    parser.add_argument("--device", type=str, default="auto", help="Device selection placeholder (MLX selects automatically)")
    parser.add_argument("--eval-max-batches", type=int, default=50)
    # Sampling for qualitative checks
    parser.add_argument("--sp-model", type=str, default=None, help="SentencePiece model path for sample decoding (optional)")
    parser.add_argument("--sp-meta", type=str, default=None, help="Tokenizer meta JSON path (optional)")
    parser.add_argument("--sample-prompt", type=str, default="Once upon a time,", help="Prompt used for sample generation logs")
    parser.add_argument("--sample-max-new-tokens", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    model = GPT(cfg)
    print(json.dumps({"params": count_parameters(model), "config": asdict(cfg)}, indent=2))

    # Optional tokenizer for qualitative samples
    sp: Optional[spm.SentencePieceProcessor] = None
    special_ids: Optional[dict] = None
    if args.sp_model:
        try:
            sp = spm.SentencePieceProcessor(model_file=args.sp_model)
            meta_path = Path(args.sp_meta) if args.sp_meta else Path(args.sp_model).with_suffix('.meta.json')
            with meta_path.open('r', encoding='utf-8') as f:
                meta = json.load(f)
            st = meta.get('special_tokens', {})
            special_ids = {
                'pad': st['PAD']['id'],
                'unk': st['UNK']['id'],
                'bos': st['BOS']['id'],
                'eos': st['EOS']['id'],
            }
        except Exception as e:
            print(f"Warning: failed to load tokenizer for samples: {e}")
            sp = None
            special_ids = None

    train_loader = create_loader(
        args.data_dir,
        "train",
        target_batch_tokens=args.batch_tokens,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    val_loader = create_loader(
        args.data_dir,
        "val",
        target_batch_tokens=args.batch_tokens,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    sched = build_scheduler(base_lr=args.lr, warmup_steps=args.warmup_steps, total_steps=args.steps, scheduler=args.scheduler)

    def loss_fn(m: GPT, x: mx.array, y: mx.array) -> mx.array:
        logits = m(x)
        return cross_entropy_loss(logits, y)

    value_and_grad = nn.value_and_grad(model, loss_fn)

    best_ppl = float("inf")
    best_meta = None

    # Simple metadata-only resume
    if args.resume_from:
        try:
            with Path(args.resume_from).open("r", encoding="utf-8") as f:
                prev = json.load(f)
            print(json.dumps({"resumed_from": prev.get("checkpoint", args.resume_from)}, indent=2))
        except Exception as e:
            print(f"Warning: failed to read resume checkpoint metadata: {e}")

    pbar = tqdm(range(args.steps), desc="Training", unit="step")
    for step in pbar:
        batch_np = next(iter(train_loader))  # simple iteration; loader creates new iterator each call
        if batch_np.shape[1] < 2:
            continue
        x = mx.array(batch_np[:, :-1])
        y = mx.array(batch_np[:, 1:])

        lr = sched(step)
        optimizer.learning_rate = lr

        loss, grads = value_and_grad(model, x, y)
        grads = clip_grads(grads, args.grad_clip)
        optimizer.update(model, grads)

        pbar.set_postfix({"loss": f"{float(loss.item()):.4f}", "lr": f"{lr:.2e}"})

        if (step + 1) % args.eval_interval == 0 or step == 0:
            metrics = evaluate_model(model, val_loader, max_batches=args.eval_max_batches)
            cur_ppl = metrics.get("perplexity", float("inf"))
            print(json.dumps({"step": step + 1, "val": metrics}, indent=2))
            # Print a qualitative sample if tokenizer is available
            if sp is not None and special_ids is not None:
                sample_text = _sample_greedy(
                    model,
                    sp,
                    special_ids,
                    args.sample_prompt,
                    cfg.seq_len,
                    args.sample_max_new_tokens,
                )
                print("\n=== Sample ===")
                print(sample_text)
                print("=== End Sample ===\n")
            # Save best metadata
            if cur_ppl < best_ppl:
                best_ppl = cur_ppl
                best_meta = {
                    "step": step + 1,
                    "val": metrics,
                    "config": asdict(cfg),
                }
                with (checkpoints_dir / "best.json").open("w", encoding="utf-8") as f:
                    json.dump(best_meta, f, indent=2)

        # Save last metadata periodically
        if (step + 1) % args.save_interval == 0 or (step + 1) == args.steps:
            last_meta = {
                "step": step + 1,
                "config": asdict(cfg),
                # Ensure JSON-serializable value
                "learning_rate": float(lr),
            }
            with (checkpoints_dir / "last.json").open("w", encoding="utf-8") as f:
                json.dump(last_meta, f, indent=2)

    # Save a minimal training summary
    with (checkpoints_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"steps": args.steps, "config": asdict(cfg)}, f, indent=2)


if __name__ == "__main__":
    main()

