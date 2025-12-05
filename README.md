# Standard Attention LLM

Minimal GPT-style language model.

## Quick Start

```bash
pip install torch torchtune transformers datasets

python train.py
```

## Files

- `model.py` - GPT model (attention + RoPE + SwiGLU)
- `train.py` - Training with SmolLM dataset
- `config.py` - Configuration

## Options

```bash
python train.py --max_steps 500 --batch_size 8 --num_layers 4
```