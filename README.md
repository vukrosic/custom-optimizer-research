# Custom Optimizer Research

Check the YouTube video explaining this repo and research:

**[🎥 YouTube](https://youtu.be/sa52y23K1T4)** | **[💬 Discord](https://discord.gg/6AbXGpKTwN)**

Research on optimizers for neural networks. Explores how and why Muon beats other optimizers like AdamW.

Main goal is understanding how the current best optimizers work, and to use that understanding to come up with new ideas.

Experiments below are just one way of understanding it, but since this research in the early stages, we will make many changes. Your suggestions are welcome.

Some ideas explored: how geometric constraints (Stiefel, Oblique, Symplectic, etc.) affect weight matrix transformations and training dynamics.

## Key Findings

| Optimizer | MNIST Acc | Characteristic |
|-----------|-----------|----------------|
| **AdamW** | 97.69% | Reliable baseline |
| **Oblique** | 97.64% | Unit-norm columns, lowest loss |
| **Muon** | 96.96% | Fastest early convergence |
| **L1-Stiefel** | 96.95% | Sparse orthogonal updates |

## Full Optimizer Comparison (5 epochs)

| Optimizer | Loss | Accuracy | Notes |
|-----------|------|----------|-------|
| **AdamW** | 0.043 | **97.87%** | Best accuracy |
| **Block-Stiefel** | 0.045 | 97.80% | Block-orthogonal for multi-head attention |
| **SL-Muon** | 0.055 | 97.48% | Volume-preserving (det=1) |
| **Oblique** | **0.033** | 97.28% | Lowest loss, unit-norm columns |
| **L1-Stiefel** | 0.113 | 97.07% | Sparse + orthogonal |
| **SGD** | 0.099 | 96.91% | Baseline |
| **Muon** | 0.082 | 96.90% | Newton-Schulz gradient orthogonalization |
| **Symplectic** | 0.048 | 96.66% | Energy-preserving for physics |
| Doubly-Stochastic | 1.60 | 39.21% | ⚠️ Designed for permutation learning |
| Grassmannian | 125.4 | 11.35% | ❌ Bug - needs fix |

## Quick Start

```bash
pip install -r requirements.txt

# Test a single optimizer
python mnist/test_optimizer.py --optimizer muon --epochs 2

# Run all optimizers
python mnist/test_optimizer.py --optimizer all --epochs 5
```

## Structure

- `optimizers/` - Custom optimizer implementations (Muon, Oblique, SL-Muon, Symplectic, etc.)
- `mnist/` - MNIST experiments and analysis
- `llm/` - LLM-scale experiments
- `research_paper/` - Motivation, hypotheses, and paper draft

## Research Docs

- [Motivation](research_paper/motivation.md)
- [Research Questions](research_paper/research_questions_and_hypothesis.md) 
- [MNIST Research Report](mnist/results/RESEARCH_REPORT.md)    