import argparse
import json
from pathlib import Path
from typing import Dict, List

import sentencepiece as spm


def parse_special_tokens(tokens_arg: str) -> Dict[str, str]:
    parts = [t.strip() for t in tokens_arg.split(",") if t.strip()]
    expected = {"PAD", "BOS", "EOS", "UNK"}
    normalized = [p.strip("[]") for p in parts]
    mapping = {p: f"[{p}]" for p in normalized}
    if set(normalized) != expected:
        raise ValueError(
            "--special-tokens must contain exactly: [PAD],[BOS],[EOS],[UNK]"
        )
    return {
        "pad": mapping["PAD"],
        "bos": mapping["BOS"],
        "eos": mapping["EOS"],
        "unk": mapping["UNK"],
    }


def train_sentencepiece_bpe(
    input_path: Path,
    model_prefix: Path,
    vocab_size: int = 1024,
    special_tokens: str = "[PAD],[BOS],[EOS],[UNK]",
    character_coverage: float = 1.0,
    model_type: str = "bpe",
) -> Dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    tokens = parse_special_tokens(special_tokens)

    trainer_cmd: List[str] = [
        f"--input={str(input_path)}",
        f"--model_prefix={str(model_prefix)}",
        f"--vocab_size={int(vocab_size)}",
        f"--model_type={model_type}",
        f"--character_coverage={character_coverage}",
        # Assign IDs explicitly so downstream code can rely on stable IDs
        f"--pad_id=0",
        f"--unk_id=1",
        f"--bos_id=2",
        f"--eos_id=3",
        f"--pad_piece={tokens['pad']}",
        f"--unk_piece={tokens['unk']}",
        f"--bos_piece={tokens['bos']}",
        f"--eos_piece={tokens['eos']}",
        # Improves robustness on unusual bytes
        f"--byte_fallback=true",
    ]

    spm.SentencePieceTrainer.Train(" ".join(trainer_cmd))

    model_file = Path(f"{str(model_prefix)}.model")
    vocab_file = Path(f"{str(model_prefix)}.vocab")

    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    ids = {
        "pad_id": sp.piece_to_id(tokens["pad"]),
        "unk_id": sp.piece_to_id(tokens["unk"]),
        "bos_id": sp.piece_to_id(tokens["bos"]),
        "eos_id": sp.piece_to_id(tokens["eos"]),
        "vocab_size": sp.vocab_size(),
    }

    meta = {
        "model_file": str(model_file),
        "vocab_file": str(vocab_file),
        "vocab_size": ids["vocab_size"],
        "special_tokens": {
            "PAD": {"token": tokens["pad"], "id": ids["pad_id"]},
            "UNK": {"token": tokens["unk"], "id": ids["unk_id"]},
            "BOS": {"token": tokens["bos"], "id": ids["bos_id"]},
            "EOS": {"token": tokens["eos"], "id": ids["eos_id"]},
        },
    }

    meta_path = Path(f"{str(model_prefix)}.meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece BPE tokenizer with reserved special tokens.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the training text file (e.g., data/raw/train.txt).",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        required=True,
        help="Output prefix for the SentencePiece model and vocab files.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1024,
        help="Vocabulary size (includes special tokens).",
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        default="[PAD],[BOS],[EOS],[UNK]",
        help="Comma-separated list of special tokens exactly as: [PAD],[BOS],[EOS],[UNK]",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=1.0,
        help="Amount of characters covered by the model, 1.0 for ASCII/English-heavy corpora.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram"],
        help="SentencePiece model type.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = train_sentencepiece_bpe(
        input_path=Path(args.input),
        model_prefix=Path(args.model_prefix),
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()


