from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import sentencepiece as spm

import mlx.core as mx

from src.model.gpt_mlx import GPT, ModelConfig


def apply_repetition_penalty(logits_np: np.ndarray, generated: List[int], penalty: float) -> np.ndarray:
    if penalty <= 1.0 or len(generated) == 0:
        return logits_np
    logits = logits_np.copy()
    unique_ids = set(generated)
    for tid in unique_ids:
        logits[tid] /= penalty
    return logits


def top_k_top_p_filtering(probs: np.ndarray, top_k: int = 0, top_p: float = 1.0) -> np.ndarray:
    # Returns a masked probability distribution that re-normalizes to 1
    p = probs.copy()
    V = p.shape[0]

    if top_k > 0 and top_k < V:
        # Keep only top_k probs
        idx = np.argpartition(-p, top_k)[:top_k]
        mask = np.zeros_like(p, dtype=bool)
        mask[idx] = True
        p[~mask] = 0.0

    if top_p < 1.0:
        # Sort by prob descending and keep smallest set whose sum >= top_p
        sorted_idx = np.argsort(-p)
        sorted_p = p[sorted_idx]
        cumsum = np.cumsum(sorted_p)
        cutoff = np.searchsorted(cumsum, top_p)
        keep = sorted_idx[: max(1, cutoff + 1)]
        mask = np.zeros_like(p, dtype=bool)
        mask[keep] = True
        p[~mask] = 0.0

    s = p.sum()
    if s <= 0:
        # fallback to uniform over all tokens
        p = np.ones_like(p) / float(V)
    else:
        p = p / s
    return p


def sample_tokens(
    model: GPT,
    sp: spm.SentencePieceProcessor,
    special_ids: dict,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    seq_len: int = 256,
    seed: int = 42,
) -> str:
    rng = np.random.default_rng(seed)
    input_ids: List[int] = [special_ids["bos"]] + sp.encode(prompt, out_type=int)
    input_ids = input_ids[-seq_len:]

    generated: List[int] = []
    for _ in range(max_new_tokens):
        ctx = input_ids[-seq_len:]
        logits = model(mx.array([ctx]))  # (1, T, V)
        last = logits[0, -1, :]
        # Convert to numpy for sampling
        last_np = np.array(last)
        if temperature > 0:
            last_np = last_np / max(1e-6, temperature)
        last_np = apply_repetition_penalty(last_np, generated, repetition_penalty)
        # Softmax
        exp = np.exp(last_np - last_np.max())
        probs = exp / np.sum(exp)
        probs = top_k_top_p_filtering(probs, top_k=top_k, top_p=top_p)

        next_token = int(rng.choice(len(probs), p=probs))
        input_ids.append(next_token)
        if next_token == special_ids["eos"]:
            break
        generated.append(next_token)

    filtered = [t for t in generated if t not in (special_ids["bos"], special_ids["eos"], special_ids["pad"]) ]
    try:
        return sp.decode(filtered)
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from the GPT model")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/best.json", help="Path to checkpoint metadata JSON")
    parser.add_argument("--sp-model", type=str, required=True, help="SentencePiece model path")
    parser.add_argument("--sp-meta", type=str, default=None, help="SentencePiece meta JSON path")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    # Model config overrides (when no checkpoint meta is available)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=7)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load tokenizer and special IDs
    sp = spm.SentencePieceProcessor(model_file=args.sp_model)
    meta_path = Path(args.sp_meta) if args.sp_meta else Path(args.sp_model).with_suffix(".meta.json")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    st = meta.get("special_tokens", {})
    special_ids = {
        "pad": st["PAD"]["id"],
        "unk": st["UNK"]["id"],
        "bos": st["BOS"]["id"],
        "eos": st["EOS"]["id"],
    }

    # Configure model from checkpoint metadata if present
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        try:
            with ckpt.open("r", encoding="utf-8") as f:
                ckpt_meta = json.load(f)
            c = ckpt_meta.get("config") or {}
            cfg = ModelConfig(
                vocab_size=c.get("vocab_size", cfg.vocab_size),
                seq_len=c.get("seq_len", cfg.seq_len),
                d_model=c.get("d_model", cfg.d_model),
                n_layers=c.get("n_layers", cfg.n_layers),
                n_heads=c.get("n_heads", cfg.n_heads),
                dropout=c.get("dropout", cfg.dropout),
            )
        except Exception as e:
            print(f"Warning: failed to read checkpoint meta: {e}")

    model = GPT(cfg)
    # Note: actual weight loading is not yet implemented; generation uses current model parameters

    text = sample_tokens(
        model,
        sp,
        special_ids,
        args.prompt,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seq_len=cfg.seq_len,
        seed=args.seed,
    )
    print(text)


if __name__ == "__main__":
    main()


