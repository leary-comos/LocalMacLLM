# LocalMacLLM 🚀

> **Train a 1.5M parameter GPT-style language model on your MacBook Pro in under 10 minutes**

A complete, end-to-end implementation of a tiny language model that demonstrates the full pipeline from raw text data to a working, interactive LLM. Built specifically for Apple Silicon Macs using MLX.

## 🎯 Project Overview

This is an experimental project to learn the basic construction of Large Language Models (LLMs) and understand the complete process of how one is built from scratch. The goal is to create a fully functional, albeit small, language model that can generate coherent text and demonstrate the fundamental principles of modern LLM architecture.

### Inspiration

This project was inspired by Sean Goedecke's excellent guide ["Training a Model on a MacBook Pro (M1 Pro)"](https://www.seangoedecke.com/model-on-a-mbp/), which demonstrates that you can train meaningful language models on consumer hardware. The guide shows that a GPT-style transformer with roughly 1.8 million parameters, trained on about 20 million TinyStories tokens, can reach a perplexity of around 9.6 and produce simple, coherent short stories in about five minutes on an M1 Pro.

### What You'll Learn

- **Complete LLM Pipeline**: From raw text data to interactive model
- **Transformer Architecture**: GPT-style model with attention mechanisms
- **Tokenization**: SentencePiece BPE tokenizer training and usage
- **Training Process**: Optimizers, learning rate scheduling, evaluation
- **Apple MLX Framework**: Native machine learning on Apple Silicon
- **Practical Constraints**: Working within memory and time limitations

## 🏗️ Architecture

- **Model**: GPT-style transformer with ~1.5M parameters
- **Framework**: Apple MLX (optimized for Apple Silicon)
- **Dataset**: TinyStories (~20M tokens of simple stories)
- **Tokenizer**: SentencePiece BPE with 1,024 vocabulary size
- **Context Length**: 256 tokens
- **Training Time**: 5-10 minutes on M1 Pro

### Model Specifications

```
Parameters: ~1.5M
- d_model: 128
- n_layers: 7  
- n_heads: 4
- vocab_size: 1024
- seq_len: 256
```

## 🚀 Quick Start

### Prerequisites

- **macOS ≥ 13.5**
- **Apple Silicon** (M1/M2/M3). Tested on M1 Pro with 16GB RAM
- **Python ≥ 3.8** (3.13 recommended)

### Installation & Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd LocalMacLLM

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Complete Training Pipeline

```bash
# 1. Download and prepare TinyStories dataset
python -m src.data.download_tinystories \
  --out data/raw \
  --val-ratio 0.01 \
  --test-ratio 0.01

# 2. Train SentencePiece tokenizer
python -m src.tokenizer.train_sentencepiece \
  --input data/raw/train.txt \
  --vocab-size 1024 \
  --model-prefix artifacts/tokenizer/sp_bpe_1k \
  --special-tokens "[PAD],[BOS],[EOS],[UNK]"

# 3. Preprocess and pack sequences
python -m src.preprocess.pack_sequences \
  --input data/raw \
  --sp-model artifacts/tokenizer/sp_bpe_1k.model \
  --seq-len 256 \
  --out data/packed

# 4. Train the model (5-10 minutes)
python -m src.train \
  --data-dir data/packed \
  --checkpoints-dir artifacts/checkpoints \
  --d-model 128 --n-layers 7 --n-heads 4 \
  --vocab-size 1024 --seq-len 256 \
  --lr 3e-4 --warmup-steps 200 --scheduler cosine \
  --batch-tokens 8192 --steps 2000 \
  --eval-interval 200 --seed 42

# 5. Evaluate the model
python -m src.eval \
  --data-dir data/packed \
  --checkpoint artifacts/checkpoints/best.json

# 6. Generate stories!
python -m src.generate \
  --checkpoint artifacts/checkpoints/best.json \
  --sp-model artifacts/tokenizer/sp_bpe_1k.model \
  --prompt "Once upon a time, there was a small robot who" \
  --max-new-tokens 120 \
  --temperature 0.9 \
  --top-k 40 \
  --top-p 0.95
```

### Quick Training Script

For a faster demo, use the convenience script:

```bash
./scripts/run_quick_train.sh
```

## 📁 Project Structure

```
LocalMacLLM/
├── src/
│   ├── data/
│   │   └── download_tinystories.py      # Dataset downloader
│   ├── tokenizer/
│   │   └── train_sentencepiece.py       # BPE tokenizer training
│   ├── preprocess/
│   │   └── pack_sequences.py            # Sequence packing
│   ├── datasets/
│   │   └── tinystories_dataset.py       # Memory-mapped loaders
│   ├── model/
│   │   └── gpt_mlx.py                   # GPT model implementation
│   ├── train.py                         # Training loop
│   ├── eval.py                          # Evaluation
│   └── generate.py                      # Text generation
├── scripts/
│   └── run_quick_train.sh               # Quick training script
├── artifacts/                           # Tokenizer & checkpoints
├── data/                               # Raw & processed datasets
└── tests/                              # Unit tests
```

## ⚙️ Configuration

### Default Settings

- **Tokenizer**: SentencePiece BPE, vocab=1024
- **Special Tokens**: `[PAD]`, `[BOS]`, `[EOS]`, `[UNK]`
- **Context Length**: 256 tokens
- **Batch Size**: ~8k tokens per optimization step
- **Optimizer**: AdamW with learning rate 3e-4
- **Schedule**: 200 warmup steps, cosine decay
- **Regularization**: Weight decay 0.1, gradient clipping 1.0

### Training Parameters

- **Evaluation**: Every 200 steps
- **Checkpointing**: Save best model on perplexity improvement
- **Sampling**: Periodic qualitative samples during training

## 🎮 Usage Examples

### Generate Stories

```bash
# Simple story generation
python -m src.generate \
  --checkpoint artifacts/checkpoints/best.json \
  --sp-model artifacts/tokenizer/sp_bpe_1k.model \
  --prompt "The little cat" \
  --max-new-tokens 100 \
  --temperature 0.8

# More creative generation
python -m src.generate \
  --checkpoint artifacts/checkpoints/best.json \
  --sp-model artifacts/tokenizer/sp_bpe_1k.model \
  --prompt "A robot named" \
  --max-new-tokens 150 \
  --temperature 1.0 \
  --top-k 50 \
  --top-p 0.9
```

### Evaluate Performance

```bash
# Check validation perplexity
python -m src.eval \
  --data-dir data/packed \
  --checkpoint artifacts/checkpoints/best.json
```

## 🔧 Tips & Optimization

### Performance Tips

- **Close heavy applications** before training to free memory
- **Reduce steps** for faster demo: `--steps 1200`
- **Smaller batches** for less memory: `--batch-tokens 4096`
- **Extend training** for better quality: `--steps 3000-5000`

### Memory Management

The model is designed to fit comfortably in 16GB RAM. If you encounter memory issues:
- Reduce `--batch-tokens`
- Close other applications
- Use a smaller model (reduce `--d-model` or `--n-layers`)

## 🐛 Troubleshooting

### Common Issues

**MLX Import Error**
```bash
# Ensure macOS ≥ 13.5 and Apple Silicon
pip install --upgrade mlx
```

**PyArrow Errors**
```bash
pip install --upgrade pyarrow
```

**Tokenizer Not Found**
```bash
# Re-run tokenizer training
python -m src.tokenizer.train_sentencepiece \
  --input data/raw/train.txt \
  --vocab-size 1024 \
  --model-prefix artifacts/tokenizer/sp_bpe_1k
```

**Memory Issues**
- Reduce `--batch-tokens` to 4096 or 2048
- Close other applications
- Ensure you have at least 8GB free RAM

## 📊 Expected Results

After training for 5-10 minutes, you should see:
- **Perplexity**: ~9-15 on validation set
- **Sample Quality**: Simple but coherent short stories
- **Training Loss**: Decreasing from ~7.5 to ~2-3

The model will generate text that, while not perfect, demonstrates understanding of basic story structure and language patterns.

## 🎓 Educational Value

This project serves as a practical introduction to:

1. **Transformer Architecture**: Understanding attention mechanisms and the GPT architecture
2. **Training Dynamics**: Learning rate scheduling, gradient clipping, evaluation
3. **Data Pipeline**: From raw text to tokenized, batched training data
4. **Hardware Constraints**: Working within memory and compute limitations
5. **Production Considerations**: Checkpointing, evaluation, and deployment

## 🤝 Contributing

This is primarily an educational project, but contributions are welcome! Areas for improvement:

- Better model architectures
- More efficient training techniques
- Additional evaluation metrics
- Documentation improvements
- Bug fixes and optimizations

## 📄 License

MIT

This project is for educational and experimental use. Feel free to use, modify, and learn from it!

## Acknowledgments

- **Sean Goedecke** for the inspiring guide that started this project
- **Apple MLX Team** for the excellent ML framework
- **TinyStories Dataset** creators for the perfect training data
- **SentencePiece** team for the robust tokenization library

---
