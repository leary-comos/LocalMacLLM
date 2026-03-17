from __future__ import annotations

import glob
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _discover_shards(packed_dir: Path, split: str) -> List[Path]:
    pattern = str(packed_dir / f"{split}_*.npy")
    shard_paths = sorted(Path(p) for p in glob.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found for split='{split}' under {packed_dir}")
    return shard_paths


@dataclass
class ShardInfo:
    path: Path
    num_blocks: int
    seq_len: int
    dtype: np.dtype


class ShardedBlockDataset:
    """
    Memory-mapped access to packed blocks across multiple shard files.

    Each shard is a .npy array with shape (num_blocks, seq_len), dtype uint16/uint32.
    """

    def __init__(self, packed_dir: str | Path, split: str) -> None:
        self.packed_dir = Path(packed_dir)
        self.split = split
        self._shard_paths: List[Path] = _discover_shards(self.packed_dir, split)
        self._shards: List[np.memmap] = []
        self._shard_infos: List[ShardInfo] = []

        for path in self._shard_paths:
            arr = np.load(path, mmap_mode="r")  # memmap
            if arr.ndim != 2:
                raise ValueError(f"Shard {path} must be 2D (num_blocks, seq_len)")
            num_blocks, seq_len = arr.shape
            self._shards.append(arr)
            self._shard_infos.append(
                ShardInfo(path=path, num_blocks=num_blocks, seq_len=seq_len, dtype=arr.dtype)
            )

        # Validate consistent seq_len and dtype across shards
        seq_lens = {info.seq_len for info in self._shard_infos}
        dtypes = {str(info.dtype) for info in self._shard_infos}
        if len(seq_lens) != 1:
            raise ValueError(f"Inconsistent seq_len across shards: {seq_lens}")
        if len(dtypes) != 1:
            raise ValueError(f"Inconsistent dtype across shards: {dtypes}")

        self._seq_len: int = next(iter(seq_lens))
        self._dtype: np.dtype = self._shards[0].dtype

        # Prefix sums to translate global block index → (shard_id, local_index)
        self._cumulative_blocks: List[int] = [0]
        for info in self._shard_infos:
            self._cumulative_blocks.append(self._cumulative_blocks[-1] + info.num_blocks)

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def num_blocks(self) -> int:
        return self._cumulative_blocks[-1]

    def _locate(self, global_block_index: int) -> Tuple[int, int]:
        if not (0 <= global_block_index < self.num_blocks):
            raise IndexError(f"Block index {global_block_index} out of range 0..{self.num_blocks-1}")
        # Binary search over cumulative boundaries
        lo, hi = 0, len(self._cumulative_blocks) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cumulative_blocks[mid + 1] <= global_block_index:
                lo = mid + 1
            else:
                hi = mid
        shard_id = lo
        local_index = global_block_index - self._cumulative_blocks[shard_id]
        return shard_id, local_index

    def __getitem__(self, idx: int) -> np.ndarray:
        shard_id, local_index = self._locate(idx)
        return np.asarray(self._shards[shard_id][local_index], copy=True)

    def iter_indices(self, shuffle: bool, seed: int) -> Iterable[int]:
        indices = list(range(self.num_blocks))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(indices)
        return indices


class BatchLoader:
    """
    Simple NumPy batch loader yielding (batch_size, seq_len) arrays.

    Batch size is derived from target_batch_tokens // seq_len unless explicitly provided.
    """

    def __init__(
        self,
        dataset: ShardedBlockDataset,
        *,
        target_batch_tokens: int = 8192,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        self.dataset = dataset
        self.seq_len = dataset.seq_len
        self.batch_size = batch_size or max(1, int(target_batch_tokens // self.seq_len))
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        batch_size = self.batch_size
        batch: List[np.ndarray] = []
        for idx in self.dataset.iter_indices(shuffle=self.shuffle, seed=self.seed):
            batch.append(self.dataset[idx])
            if len(batch) == batch_size:
                yield np.stack(batch, axis=0)
                batch = []
        if batch and not self.drop_last:
            yield np.stack(batch, axis=0)


def create_loader(
    packed_dir: str | Path,
    split: str,
    *,
    target_batch_tokens: int = 8192,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = True,
) -> BatchLoader:
    dataset = ShardedBlockDataset(packed_dir=packed_dir, split=split)
    return BatchLoader(
        dataset,
        target_batch_tokens=target_batch_tokens,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def create_loaders(
    packed_dir: str | Path,
    splits: Sequence[str] = ("train", "val", "test"),
    *,
    target_batch_tokens: int = 8192,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = True,
) -> dict[str, BatchLoader]:
    """Convenience factory returning loaders for the given splits."""
    return {
        split: create_loader(
            packed_dir=packed_dir,
            split=split,
            target_batch_tokens=target_batch_tokens,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        for split in splits
    }


