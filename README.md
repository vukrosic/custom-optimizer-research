# Custom Optimizer Research

Check the YouTube video explaining this repo and research:

**<a href="https://youtu.be/sa52y23K1T4" target="_blank">🎥 YouTube</a>** | **[💬 Discord](https://discord.gg/6AbXGpKTwN)**

Research on optimizers for neural networks. Explores how and why Muon beats other optimizers like AdamW.

Main goal is understanding how the current best optimizers work, and to use that understanding to come up with new ideas.

Experiments below are just one way of understanding it, but since this research in the early stages, we will make many changes. Your suggestions are welcome.

Some ideas explored: how geometric constraints (Stiefel, Oblique, Symplectic, etc.) affect weight matrix transformations and training dynamics.

## Key Findings

Comprehensive experiments comparing optimization algorithms on large language models, focusing on gradient rank preservation and convergence dynamics.

**Current Focus**: Understanding how Muon and geometric manifold constraints affect LLM training dynamics.

## Optimizer Characteristics

| Optimizer | Key Feature |
|-----------|-------------|
| **AdamW** | Reliable baseline with weight decay |
| **Muon** | Newton-Schulz gradient orthogonalization |
| **Oblique** | Unit-norm columns constraint |
| **L1-Stiefel** | Sparse orthogonal updates |
| **Block-Stiefel** | Block-orthogonal for multi-head attention |
| **SL-Muon** | Volume-preserving (det=1) |
| **Symplectic** | Energy-preserving for physics-inspired architectures |

## Quick Start

```bash
pip install -r requirements.txt

# Run LLM gradient rank experiment
python -m llm.experiments.llm_gradient_rank_experiment --max_steps 200

# Run spectral dynamics analysis
python -m llm.experiments.llm_spectral_dynamics_experiment --optimizer muon

# Run final comparison
python -m llm.experiments.llm_final_comparison --max_steps 200
```

## Structure

- `optimizers/` - Custom optimizer implementations (Muon, Oblique, SL-Muon, Symplectic, etc.)
- `llm/` - Large language model experiments and analysis
  - `llm/common/` - Shared components (data, metrics, models)
  - `llm/experiments/` - Comprehensive optimizer experiments
  - `llm/configs/` - Model and training configurations
- `research_paper/` - Motivation, hypotheses, and paper draft

## Research Docs

- [Motivation](research_paper/motivation.md)
- [Research Questions](research_paper/research_questions_and_hypothesis.md)
- [LLM Experiments](llm/experiments.md)