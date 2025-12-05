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

# Run baseline (Muon for 2D, Adam for others)
python experiments/run_experiments.py --exp baseline

# Run sphere constraint comparison
python experiments/run_experiments.py --exp sphere_constraint

# Run both main experiments
python experiments/run_experiments.py --exp baseline --exp sphere_constraint

# Run all experiments
python experiments/run_experiments.py --all