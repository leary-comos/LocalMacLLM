import json
from pathlib import Path

import pytest  # type: ignore

try:
    import sentencepiece as spm
except Exception:  # pragma: no cover
    spm = None


TOKENIZER_PREFIX = Path("artifacts/tokenizer/sp_bpe_1k")
MODEL_PATH = TOKENIZER_PREFIX.with_suffix(".model")
META_PATH = TOKENIZER_PREFIX.with_suffix(".meta.json")


def require_trained_tokenizer():
    if spm is None:
        pytest.skip("sentencepiece not installed in this environment")
    if not MODEL_PATH.exists() or not META_PATH.exists():
        pytest.skip(
            "Tokenizer artifacts not found. Train tokenizer first via src.tokenizer.train_sentencepiece"
        )


def load_sp_and_meta():
    require_trained_tokenizer()
    sp = spm.SentencePieceProcessor(model_file=str(MODEL_PATH))
    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return sp, meta


def test_special_tokens_present_and_ids():
    sp, meta = load_sp_and_meta()

    specials = meta["special_tokens"]
    assert "PAD" in specials and "UNK" in specials and "BOS" in specials and "EOS" in specials

    assert specials["PAD"]["id"] == 0
    assert specials["UNK"]["id"] == 1
    assert specials["BOS"]["id"] == 2
    assert specials["EOS"]["id"] == 3

    # Also verify SentencePiece mapping agrees
    assert sp.id_to_piece(0) == specials["PAD"]["token"]
    assert sp.id_to_piece(1) == specials["UNK"]["token"]
    assert sp.id_to_piece(2) == specials["BOS"]["token"]
    assert sp.id_to_piece(3) == specials["EOS"]["token"]


def test_encode_decode_round_trip():
    sp, _ = load_sp_and_meta()
    text = "Once upon a time, a small robot learned to tell stories."
    ids = sp.encode(text, out_type=int)
    assert isinstance(ids, list) and len(ids) > 0
    decoded = sp.decode(ids)
    assert isinstance(decoded, str) and len(decoded) > 0
    # SentencePiece preserves content; minor whitespace normalization can occur
    assert decoded.strip() == text.strip()


