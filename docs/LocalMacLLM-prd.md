# LocalMacLLM PRD: 1.5M-Parameter GPT-Style LLM on MacBook Pro (M1 Pro)

## Introduction / Overview

Build and train a small GPT-style language model (~1.5 million parameters) locally on a MacBook Pro M1 Pro (16GB RAM, 1TB storage) using Apple's MLX framework. The model will be trained on roughly 20 million tokens from TinyStories to achieve validation perplexity around 10 and generate short, coherent stories. Target quick iteration runs of ~5–10 minutes for a demo-quality model, with an option to extend training to ~15–30 minutes for improved quality. This PRD is written for a junior developer and specifies explicit steps, requirements, and acceptance criteria.

References:
- Guidance inspired by "Model on a MacBook Pro (M1 Pro)" which reports ~1.8M-parameter GPT trained on ~20M TinyStories tokens achieving ~9.6 perplexity in minutes. See [Model on a MacBook Pro (M1 Pro)](https://www.seangoedecke.com/model-on-a-mbp/).

## Goals

1. Provide a reproducible local training pipeline using MLX optimized for Apple Silicon.
2. Train a ~1.5M parameter GPT-style model on ~20M TinyStories tokens with validation perplexity ≤ ~10.
3. Keep wall-clock for a "quick run" within ~5–10 minutes on M1 Pro (with options for extended runs).
4. Minimize RAM footprint through small context length, small batch tokens, and efficient tokenization.
5. Deliver sampling utilities to generate short stories from prompts.
6. Save best and last checkpoints; provide simple CLI entry points for train/eval/generate.

## User Stories

- As a developer, I want a one-command setup to train a small LLM locally so I can iterate quickly without cloud costs.
- As an experimenter, I want to tweak model size, batch tokens, and steps to balance speed and quality on my Mac.
- As a hobbyist, I want to generate simple, coherent short stories offline using a locally trained model.

## Functional Requirements

1. Environment setup
   1.1. Provide commands to create and activate a Python virtual environment.
   1.2. Install dependencies: `mlx`, `datasets`, `sentencepiece`, `tqdm`.
   1.3. Document minimum OS requirement (macOS ≥ 13.5) and Apple Silicon (M1/M2/M3).

2. Data acquisition and preparation
   2.1. Use TinyStories dataset (Hugging Face `roneneldan/TinyStories`).
   2.2. Target ~20M training tokens; define train/validation/test splits (e.g., 98%/1%/1% or similar).
   2.3. Stream/download data and store locally under a project data directory.

3. Tokenization
   3.1. Train a SentencePiece BPE tokenizer with vocabulary size = 1,024 on the training subset.
   3.2. Define and reserve special tokens: PAD, BOS, EOS, and optionally UNK.
   3.3. Save tokenizer artifacts (`.model`, `.vocab`) to `artifacts/tokenizer/` and version them.
   3.4. Provide a script/command to re-train tokenizer if dataset changes.

4. Dataset preprocessing
   4.1. Tokenize the dataset using the trained SentencePiece model.
   4.2. Pack token sequences into fixed-length blocks with context length = 256, respecting BOS/EOS.
   4.3. Implement efficient storage/loading (e.g., memory-mapped arrays or shard files) for training.
   4.4. Provide data loader that yields ~8k tokens per step via dynamic batch sizing or accumulation.

5. Model architecture (GPT-style)
   5.1. Embedding: token embeddings with weight tying to the LM head; optional positional embeddings.
   5.2. Transformer blocks: n_layers = 7, d_model = 128, n_heads = 4; pre-norm; GELU activation.
   5.3. Context length = 256; dropout default ≤ 0.1 (may be disabled for fastest convergence on small data).
   5.4. Output: tied projection to vocabulary; cross-entropy loss with label smoothing = 0.0.
   5.5. Parameter budget ≈ 1.5M (see Technical Considerations for breakdown).

6. Training configuration
   6.1. Optimizer: AdamW with weight decay = 0.1; gradient clip = 1.0.
   6.2. Learning rate: 3e-4 with warmup steps = 200 and cosine decay schedule.
   6.3. Batch tokens: target ≈ 8k per optimization step (use accumulation if needed to fit RAM).
   6.4. Steps: quick run ≤ ~2,000 steps (≈16M tokens if 8k tokens/step); extended run configurable (e.g., 3,000–5,000 steps).
   6.5. Evaluation cadence: compute validation perplexity every 200 steps.
   6.6. Checkpointing: save `last` every N steps and `best` when validation perplexity improves.
   6.7. Determinism: set random seed and document any nondeterministic behaviors.

7. Evaluation and monitoring
   7.1. Report validation loss and perplexity at the configured interval.
   7.2. Print sample short stories at intervals using a fixed prompt for qualitative checks.
   7.3. Optional: TensorBoard logging (off by default to minimize overhead).

8. Inference and sampling
   8.1. Provide a `generate` CLI that loads `best` (or user-specified) checkpoint and tokenizer.
   8.2. Sampling controls: max tokens, temperature, top-k, top-p, repetition penalty.
   8.3. Output to console with option to write to file.

9. Artifacts and project structure
   9.1. Use conventional directories: `src/`, `scripts/`, `artifacts/` (tokenizer, checkpoints), `data/`.
   9.2. Include a README with clear setup, training, evaluation, and generation instructions.

10. Resource constraints
   10.1. Must run within 16GB unified memory without swapping under typical conditions.
   10.2. Quick run must target ≤ ~10 minutes on M1 Pro; extended runs configurable by flags.
   10.3. Avoid excessive logging; prefer lightweight progress bars.

## Non-Goals (Out of Scope)

1. Models ≥ ~2M parameters or long context windows beyond 512 tokens.
2. Distributed training or multi-GPU support.
3. Advanced alignment methods (RLHF), instruction tuning, or large-scale pretraining beyond TinyStories.
4. Production-grade serving, quantization/export formats beyond a future optional task.
5. Cross-platform GPU support beyond Apple Silicon (MPS/PyTorch path explicitly out of scope for v1).

## Design Considerations (Optional)

- Simplicity over completeness: prioritize a fast, reproducible path to a usable tiny model.
- Weight tying reduces parameters and improves efficiency for small vocabularies.
- Short context (256) balances compute and capability for story-length outputs.
- Dynamic token batching or gradient accumulation to maintain ~8k tokens/step under memory limits.
- Minimal defaults; optional flags to scale steps or enable TensorBoard.

## Technical Considerations (Optional)

- Approximate parameter count (for d_model = 128, n_layers = 7, n_heads = 4, vocab = 1,024, context = 256, tied embeddings):
  - Token embeddings: 1,024 × 128 ≈ 131,072
  - Positional embeddings (learned): 256 × 128 ≈ 32,768
  - Per block (attention + MLP + norms), rough estimate ≈ 197,600 → 7 blocks ≈ 1,383,200
  - Final layer norm, small biases: ≈ a few hundred to a few thousand
  - Tied output head: negligible additional params compared to untied head
  - Total ≈ 1.55M (± a small margin depending on exact implementation)
- Token budget vs. steps: with ~8k tokens/step, 2,000 steps ≈ 16M tokens; 2,500 steps ≈ 20M tokens.
- Expected quality: with ~20M tokens, validation perplexity around ~9–10 is plausible for TinyStories; coherent short-story generations expected.
- Platform: MLX requires macOS ≥ 13.5; performance benefits from Apple Silicon acceleration.

## Success Metrics

1. Quick run completes in ≤ ~10 minutes on M1 Pro for the default configuration.
2. Validation perplexity ≤ ~10 after the recommended token budget (~20M tokens).
3. Generation script produces coherent, grammatically correct short stories from fixed prompts.
4. Reproducible runs: setting the same seed yields comparable validation curves and samples.
5. Project includes clear documentation and CLI entry points for train/eval/generate.

## Open Questions

1. Should we include a curated TinyStories subset for even faster experiments (e.g., 10M tokens) as an optional mode?
2. Preferred default sampling params for demo (`temperature`, `top_k`, `top_p`)?
3. Should we add optional TensorBoard logging by default or keep it opt-in to minimize overhead?
4. Any specific prompts or evaluation suites (e.g., grammar or story coherence checks) to include?
5. Future scope: add quantized export and a tiny web demo for local inference?

## Acceptance Criteria

- A developer can follow the README to:
  - Create a virtual environment and install dependencies.
  - Download/prepare TinyStories and train the SentencePiece tokenizer (vocab = 1,024).
  - Train the 1.5M-parameter GPT configuration using MLX with default quick-run settings.
  - See periodic validation perplexity and sample story outputs during training.
  - Save and load checkpoints (`best` and `last`) and generate text via a CLI script.
- On the reference hardware (M1 Pro, 16GB RAM), quick run completes within ~5–10 minutes and produces intelligible short stories.


