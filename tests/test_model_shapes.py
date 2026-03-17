from __future__ import annotations

import numpy as np

import mlx.core as mx

from src.model.gpt_mlx import GPT, ModelConfig, count_parameters


def test_forward_shapes_and_no_nans():
    cfg = ModelConfig(vocab_size=1024, seq_len=256, d_model=128, n_layers=7, n_heads=4, dropout=0.0)
    model = GPT(cfg)

    batch_size = 2
    T = 64
    tokens_np = np.random.randint(0, cfg.vocab_size, size=(batch_size, T), dtype=np.int32)
    tokens = mx.array(tokens_np)

    logits = model(tokens)
    assert logits.shape == (batch_size, T, cfg.vocab_size)

    # Ensure no NaNs
    num_nans = int(mx.sum(mx.isnan(logits)).item())
    assert num_nans == 0


def test_parameter_count_within_expected_range():
    cfg = ModelConfig(vocab_size=1024, seq_len=256, d_model=128, n_layers=7, n_heads=4, dropout=0.0)
    model = GPT(cfg)
    total = count_parameters(model)
    # Expect around ~1.5M; allow a reasonable margin for exact implementation details
    assert 1_300_000 <= total <= 1_800_000


