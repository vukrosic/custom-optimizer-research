# Blueberry LLM

**Open Superintelligence Lab** - Open research for everyone. We publish all of our research for the sake of accelerating science. Learn real AI research from a real research lab.

## Quick Start

```bash
# Install Flash Attention
pip install flash-attn --no-build-isolation

# Install other dependencies
pip install -r requirements.txt
```

**Note**: Flash Attention installation can be tricky. If it fails:
- Ensure you have CUDA 11.8+ and a compatible GPU
- Try: `pip install flash-attn==2.5.0 --no-build-isolation`
- If still failing, you will have to try running the code without it or ask AI to elp you debug it.

## About

Purpose of this repository is to research better, faster, smarter LLMs.

This repository contains cutting-edge language model experiments and architectures. We believe scientists do their best work when given freedom to explore, so this is a space for your independent research and discovery.

Fork this repository, create a new experiment in `experiments/` folder, then create a pull request to merge it back.

## Experiments

**Research Question**: Can a model dynamically choose between linear attention (GDN) and softmax attention per-token at each layer to achieve better performance than static layer assignments?

**Train Baseline (Static)**
```bash
python experiments/run_experiment.py --config baseline
```
This trains a 4-layer model: [GDN, GDN, GDN, Softmax]

**Train Dynamic Routing**
```bash
python experiments/run_experiment.py --config dynamic
```
This trains a 4-layer model: [GDN, ROUTED, ROUTED, Softmax]

**Compare Results**
```bash
python experiments/compare_experiments.py
```

**Troubleshooting**

**Out of Memory**: Reduce batch size in `experiments/config.py`:
```python
batch_size=8  # Default is 16, reduce if OOM
```

## Structure

- **`experiments/`** - Research experiments with their own documentation
- **`models/`** - Model architectures and implementations
- **`training/`** - Training scripts and utilities
- **`configs/`** - Configuration files
- **`data/`** - Data loading and preprocessing utilities
- **`utils/`** - Helper functions and utilities
- **`optimizers/`** - Custom optimizers (e.g., Muon)
- **`benchmarks/`** - Performance benchmarking tools