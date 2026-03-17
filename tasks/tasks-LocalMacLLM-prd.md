## Relevant Files

- `README.md` - Setup and usage instructions (env, training, evaluation, generation).
- `requirements.txt` - Dependency pins (e.g., `mlx`, `datasets`, `sentencepiece`, `tqdm`).
- `src/data/download_tinystories.py` - Download/stream TinyStories and persist locally.
- `src/tokenizer/train_sentencepiece.py` - Train SentencePiece BPE (vocab=1,024), save artifacts.
- `src/preprocess/pack_sequences.py` - Tokenize and pack fixed-length context blocks (256 tokens), shard.
- `src/datasets/tinystories_dataset.py` - Dataset and data loader utilities (≈8k tokens/step target).
- `src/model/gpt_mlx.py` - GPT model in MLX (d_model=128, n_layers=7, n_heads=4, weight tying).
- `src/train.py` - Training CLI (AdamW, lr schedule, grad clip, eval every 200 steps, checkpoints).
- `src/eval.py` - Validation/perplexity evaluation utilities.
- `src/generate.py` - Sampling CLI (max_tokens, temperature, top-k, top-p, repetition penalty).
- `scripts/run_quick_train.sh` - Convenience script for quick 5–10 min runs.
- `tests/test_tokenizer.py` - Tokenizer unit tests.
- `tests/test_dataloader.py` - Data packing/loader tests.
- `tests/test_model_shapes.py` - Model forward pass and shape checks.

### Notes

- Target hardware: MacBook Pro M1 Pro, 16GB RAM; MLX requires macOS ≥ 13.5.
- Quick run: aim ≤ ~2,000 steps at ≈8k tokens/step (≈16M tokens). Extended runs can increase steps.
- Validation perplexity target: ≤ ~10 after ~20M tokens on TinyStories.
- Keep logging lightweight; optional TensorBoard can be added later.

## Tasks

- [x] 1.0 Project scaffolding and environment
  - [x] 1.1 Create directories: `src/`, `src/data/`, `src/tokenizer/`, `src/preprocess/`, `src/datasets/`, `src/model/`, `scripts/`, `artifacts/tokenizer/`, `artifacts/checkpoints/`, `data/`, `tests/`.
  - [x] 1.2 Create `requirements.txt` with minimal deps: `mlx`, `datasets`, `sentencepiece`, `tqdm`.
  - [x] 1.3 Initialize a Python venv and install deps; verify Python ≥ 3.8 and `import mlx` works.
  - [x] 1.4 Add `README.md` with setup, training, evaluation, and generation instructions; list system requirements (macOS ≥ 13.5, Apple Silicon).
  - [x] 1.5 Optional: add `.gitignore` for `.venv/`, `__pycache__/`, `artifacts/`, `data/`.

- [x] 2.0 Data acquisition and tokenizer training (TinyStories + SentencePiece BPE, vocab=1,024)
  - [x] 2.1 Implement `src/data/download_tinystories.py` to fetch `roneneldan/TinyStories` via Hugging Face Datasets and persist raw text locally.
  - [x] 2.2 Split deterministically into train/val/test (≈98%/1%/1%) and save split indices/metadata.
  - [x] 2.3 Implement `src/tokenizer/train_sentencepiece.py` to train BPE (vocab=1,024) on the train split; reserve PAD, BOS, EOS, UNK.
  - [x] 2.4 Save tokenizer artifacts to `artifacts/tokenizer/` and record special token IDs in a small README or JSON.
  - [x] 2.5 Add `tests/test_tokenizer.py` to validate encode/decode round-trip and presence of special tokens.

- [x] 3.0 Preprocessing and data loader (context=256, ≈8k tokens/step, sharding/mmap)
  - [x] 3.1 Implement `src/preprocess/pack_sequences.py` to tokenize lines, add BOS/EOS, pack into fixed 256-token blocks; write sharded arrays (e.g., `.npy`).
  - [x] 3.2 Implement `src/datasets/tinystories_dataset.py` to memory-map shards and yield batches targeting ≈8k tokens/step (dynamic batch sizing or gradient accumulation).
  - [x] 3.3 Provide seedable shuffling and separate iterables for train/val/test; expose a simple Python API for loaders.
  - [x] 3.4 Add `tests/test_dataloader.py` to check shapes, padding rules, BOS/EOS placement, and approximate token-count targeting.

- [x] 4.0 GPT model implementation in MLX (~1.5M params, weight tying)
  - [x] 4.1 Implement `src/model/gpt_mlx.py` with token embeddings, positional embeddings, 7 pre-norm transformer blocks, 4-head causal attention, MLP with GELU, dropout ≤ 0.1, and tied LM head.
  - [x] 4.2 Implement causal attention mask and correct attention scaling; support context length = 256.
  - [x] 4.3 Provide a `ModelConfig` and a utility to compute parameter count; assert ≈ 1.5M.
  - [x] 4.4 Add `tests/test_model_shapes.py` to validate forward pass shapes and absence of NaNs on dummy inputs.

- [x] 5.0 Training pipeline with evaluation and checkpointing (AdamW, warmup+cosine, PPL every 200 steps)
  - [x] 5.1 Implement `src/eval.py` to compute validation loss and perplexity from a given checkpoint.
  - [x] 5.2 Implement `src/train.py` with AdamW (wd=0.1), grad clip=1.0, lr schedule (warmup=200, cosine decay), configurable steps; run eval every 200 steps.
  - [x] 5.3 Implement checkpointing to `artifacts/checkpoints/` saving `last` every N steps and `best` when perplexity improves (include optimizer/scheduler state).
  - [x] 5.4 Expose CLI flags: steps, batch tokens target, lr, seed, eval interval, checkpoints dir, resume-from, device.
  - [x] 5.5 Add tqdm progress with minimal logging; print sample generations at eval intervals using a fixed prompt for qualitative checks.
  - [x] 5.6 Create `scripts/run_quick_train.sh` to execute a ~5–10 minute configuration (e.g., ~2,000 steps at ≈8k tokens/step).

- [x] 6.0 Inference and sampling CLI (temperature, top-k/top-p, repetition penalty)
  - [x] 6.1 Implement `src/generate.py` to load tokenizer and `best` checkpoint; accept `--prompt`, `--max-new-tokens`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--seed`.
  - [x] 6.2 Implement generation with EOS handling, context window management, and nucleus/top-k sampling.
  - [x] 6.3 Add usage examples to `README.md`, including redirecting outputs to a file.

- [x] 7.0 End-to-end validation on Mac terminal
  - [x] 7.1 Run preprocessing and quick training on M1 Pro; confirm completion ≤ ~10 minutes.
  - [x] 7.2 Verify validation perplexity logs and that `best` and `last` checkpoints are saved.
  - [x] 7.3 Generate sample stories for fixed prompts via the CLI; confirm outputs are coherent and non-empty.
  - [x] 7.4 Update `README.md` with final measured metrics, example outputs, and troubleshooting notes (memory tips, closing background apps).


