# Blueberry LLM

**Open Superintelligence Lab** - Open research for everyone. We publish all of our research for the sake of accelerating science. Learn real AI research from a real research lab.

## Quick Start

```bash
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```

## About

Purpose of this repository is to research better, faster, smarter LLMs.

This repository contains cutting-edge language model experiments and architectures. We believe scientists do their best work when given freedom to explore, so this is a space for your independent research and discovery.

Fork this repository, create a new experiment in `experiments/` folder, then create a pull request to merge it back.

## Experiments

**Research Question**: Can a model dynamically choose between linear attention (GDN) and softmax attention per-token at each layer to achieve better performance than static layer assignments?

**Train Baseline (Static)**
```bash
cd experiments
python run_experiment.py --config baseline
```
This trains a 4-layer model: [GDN, GDN, GDN, Softmax]

**Train Dynamic Routing**
```bash
python run_experiment.py --config dynamic
```
This trains a 4-layer model: [GDN, ROUTED, ROUTED, Softmax]

**Compare Results**
```bash
python compare_experiments.py
```

**Troubleshooting**

**Out of Memory**: Reduce batch size in `config.py`:
```python
batch_size=24  # Instead of 48
```

## Structure

- **`experiments/`** - Research experiments with their own documentation
- **`models/`** - Model architectures and implementations (DeepSeek, Qwen3-Next)
- **`training/`** - Training scripts and utilities
- **`configs/`** - Configuration files