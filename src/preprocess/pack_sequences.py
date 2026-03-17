import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


def load_special_ids(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    specials = meta.get("special_tokens", {})
    return {
        "pad": specials["PAD"]["id"],
        "unk": specials["UNK"]["id"],
        "bos": specials["BOS"]["id"],
        "eos": specials["EOS"]["id"],
        "vocab_size": meta.get("vocab_size"),
    }


def iter_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line


def encode_with_markers(text: str, sp: spm.SentencePieceProcessor, special_ids: dict) -> List[int]:
    ids: List[int] = [special_ids["bos"]]
    ids += sp.encode(text, out_type=int)
    ids.append(special_ids["eos"])
    return ids


def pack_and_save(
    all_ids: List[int],
    seq_len: int,
    out_dir: Path,
    split: str,
    shard_index: int,
) -> int:
    # Drop remainder that does not fill a full block
    usable = (len(all_ids) // seq_len) * seq_len
    if usable == 0:
        return 0
    blocks = np.asarray(all_ids[:usable], dtype=np.uint32).reshape(-1, seq_len)
    # Use uint16 if vocab_size <= 65535; leave as uint32 otherwise
    if blocks.max() <= np.iinfo(np.uint16).max:
        blocks = blocks.astype(np.uint16)

    out_path = out_dir / f"{split}_{shard_index:05d}.npy"
    np.save(out_path, blocks)
    return blocks.shape[0]


def process_split(
    input_file: Path,
    sp_model_path: Path,
    meta_path: Path,
    seq_len: int,
    out_dir: Path,
    blocks_per_shard: int,
) -> dict:
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    special_ids = load_special_ids(meta_path)

    buffer: List[int] = []
    shard_index = 0
    blocks_in_current_shard = 0
    total_blocks = 0
    shard_tokens: List[int] = []

    for line in tqdm(iter_lines(input_file), desc=f"Tokenizing {input_file.name}"):
        ids = encode_with_markers(line, sp, special_ids)
        buffer.extend(ids)

        # Emit blocks greedily
        while len(buffer) >= seq_len:
            shard_tokens.extend(buffer[:seq_len])
            del buffer[:seq_len]
            blocks_in_current_shard += 1

            if blocks_in_current_shard >= blocks_per_shard:
                wrote = pack_and_save(shard_tokens, seq_len, out_dir, input_file.stem, shard_index)
                total_blocks += wrote
                shard_index += 1
                blocks_in_current_shard = 0
                shard_tokens = []

    # Flush final shard
    wrote = pack_and_save(shard_tokens, seq_len, out_dir, input_file.stem, shard_index)
    total_blocks += wrote

    return {
        "split": input_file.stem,
        "blocks": total_blocks,
        "seq_len": seq_len,
        "files": [str(p) for p in sorted(out_dir.glob(f"{input_file.stem}_*.npy"))],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize and pack TinyStories lines into fixed-length blocks and save as shards.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing train/val/test .txt files, or a single .txt file.",
    )
    parser.add_argument(
        "--sp-model",
        type=str,
        required=True,
        help="Path to SentencePiece model (.model).",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Path to tokenizer meta JSON. Defaults to same prefix as --sp-model with .meta.json.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length for each packed block.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for packed shards (.npy).",
    )
    parser.add_argument(
        "--blocks-per-shard",
        type=int,
        default=8192,
        help="Number of blocks to include per shard file.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated list of split basenames to look for when --input is a directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp_model_path = Path(args.sp_model)
    if args.meta is None:
        meta_path = sp_model_path.with_suffix(".meta.json")
    else:
        meta_path = Path(args.meta)

    summaries = []
    if input_path.is_file():
        summaries.append(
            process_split(
                input_file=input_path,
                sp_model_path=sp_model_path,
                meta_path=meta_path,
                seq_len=args.seq_len,
                out_dir=out_dir,
                blocks_per_shard=args.blocks_per_shard,
            )
        )
    else:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        for split in splits:
            split_file = input_path / f"{split}.txt"
            if not split_file.exists():
                continue
            summaries.append(
                process_split(
                    input_file=split_file,
                    sp_model_path=sp_model_path,
                    meta_path=meta_path,
                    seq_len=args.seq_len,
                    out_dir=out_dir,
                    blocks_per_shard=args.blocks_per_shard,
                )
            )

    # Write an index JSON summarizing shards
    index = {s["split"]: s for s in summaries}
    with (out_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()


