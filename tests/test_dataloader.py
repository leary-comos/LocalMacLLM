from pathlib import Path

import numpy as np
import pytest

from src.datasets.tinystories_dataset import (
    ShardedBlockDataset,
    create_loader,
)


def _skip_if_no_packed(tmp_dir: Path) -> None:
    # Expect files like data/packed/train_00000.npy, etc.
    packed_dir = Path("data/packed")
    if not packed_dir.exists():
        pytest.skip("Packed data not found. Run preprocessing to generate shards under data/packed.")
    # At least one shard
    if not list(packed_dir.glob("train_*.npy")):
        pytest.skip("No train shards found under data/packed.")


def test_dataset_shapes_and_types():
    _skip_if_no_packed(Path("."))
    ds = ShardedBlockDataset("data/packed", "train")
    assert ds.num_blocks > 0
    assert ds.seq_len > 0
    arr = ds[0]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (ds.seq_len,)
    assert str(arr.dtype) in ("uint16", "uint32")


def test_loader_batching_and_token_target():
    _skip_if_no_packed(Path("."))
    ds = ShardedBlockDataset("data/packed", "train")
    target_tokens = 8192
    loader = create_loader(
        "data/packed", "train", target_batch_tokens=target_tokens, shuffle=False, drop_last=False
    )

    first_batch = next(iter(loader))
    assert first_batch.shape[1] == ds.seq_len
    # batch_size approx target_tokens/seq_len
    expected = max(1, target_tokens // ds.seq_len)
    assert first_batch.shape[0] in (expected, expected - 1, expected + 1)


