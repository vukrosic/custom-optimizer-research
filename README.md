# Standard Attention LLM

A minimal GPT-style language model with standard attention. Clean codebase for LLM experimentation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Train with custom settings
python train.py --max_steps 500 --batch_size 8 --num_layers 4
```

## Architecture

- **Attention**: Multi-head attention with RoPE (Rotary Position Embeddings)
- **FFN**: SwiGLU activation
- **Normalization**: RMSNorm (pre-norm architecture)
- **Default size**: 6 layers, 768 hidden, 12 heads (~85M params)

## Project Structure

```
├── model.py      # GPT model implementation
├── train.py      # Training script
├── config.py     # Configuration dataclasses
├── data/         # Data loading utilities
└── utils/        # Helper functions
```

## Configuration

Edit `config.py` or pass command-line arguments:

```bash
python train.py \
    --hidden_size 512 \
    --num_layers 4 \
    --num_heads 8 \
    --batch_size 32 \
    --learning_rate 1e-4
```

## Requirements

- PyTorch 2.0+
- torchtune (for RoPE)
- transformers
- datasets