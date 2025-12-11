# Custom Optimizers Research Plan

## Research Goal
Investigate how different neural network components benefit from different optimization algorithms and geometric constraints ("Modular Manifolds" concept).

## Key Hypothesis
Different parts of a neural network have different optimization needs:
- **Embeddings/Input layers**: Benefit from hypersphere constraints (prevent norm explosion/collapse)
- **Hidden layers (attention, FFN)**: Benefit from Muon optimizer (spectrally normalized updates) with optional Stiefel manifold constraints
- **Output layers**: Standard AdamW works well (unconstrained)

---

## Experiments

### 1. MNIST Newton-Schulz Iterations (`mnist_ns_experiment.py`) ✓
**Status**: Complete

Compares AdamW vs Muon on MNIST, specifically analyzing:
- Effect of 3 vs 5 Newton-Schulz iterations in Muon
- Tracks average effective rank (Shannon entropy-based) and loss
- Motivation: Understand how Muon works mechanistically

### 2. MNIST Modular Optimizer (`mnist_modular_optimizer_experiment.py`) 🆕
**Status**: New

Tests different optimizer/constraint combinations for different model parts:

| Configuration | Input Layer | Hidden Layers | Output Layer |
|--------------|-------------|---------------|--------------|
| `baseline_adamw` | AdamW | AdamW | AdamW |
| `baseline_muon` | Muon | Muon | AdamW (biases) |
| `modular_sphere` | AdamW + Sphere | Muon | Muon |
| `modular_stiefel` | Muon | Muon + Stiefel | AdamW |
| `full_modular` | AdamW + Sphere | Muon + Stiefel | AdamW |

**Metrics tracked**:
- Training/test loss and accuracy
- Layer-wise gradient effective rank
- Convergence speed

### 3. LLM Gradient Rank (`llm_gradient_rank_experiment.py`) ✓
**Status**: Complete

Large-scale experiment on 42M GPT model testing manifold constraints:
- Sphere constraint on embeddings
- Stiefel constraint on attention/FFN
- Combined constraints

---

## Manifold Constraints

### Hypersphere (for embeddings)
Each embedding vector is constrained to unit norm:
$$w \leftarrow \frac{w}{\|w\|_2}$$

### Stiefel Manifold (for weight matrices)
Matrices with orthonormal columns (all singular values = 1):
$$\text{Stiefel}(m,n) := \{W \in \mathbb{R}^{m \times n} \mid W^T W = I_n\}$$

Uses Newton-Schulz iteration to project back to manifold after each step.

### Muon Optimizer
Orthogonalizes the *gradient update* (not the weights themselves):
- Applies Newton-Schulz to the momentum buffer
- Ensures spectrally normalized learning dynamics

---

## Key Findings (so far)

1. **Hypersphere on embeddings works best** - 5.4% loss reduction on 42M GPT
2. **Stiefel has high overhead** - 35% throughput reduction, benefits don't justify cost at this scale
3. **Combining constraints doesn't stack** - Full manifold underperforms sphere-only

---

## Files Structure

```
experiments/
├── mnist_ns_experiment.py           # Muon NS iterations analysis
├── mnist_modular_optimizer_experiment.py  # Different opts for different parts
├── llm_gradient_rank_experiment.py  # Large-scale manifold constraints
├── manifold_constraints.py          # Core constraint implementations
├── modular_optimizer.py             # Modular optimizer wrapper
└── ...
```

## Understanding Experiments (New)

### 4. Spectral Dynamics (`spectral_dynamics_experiment.py`) 🆕
**Status**: New

Full SVD tracking (beyond effective rank):
- Complete singular value spectrum over training
- Condition number dynamics
- Top-k concentration ratios

### 5. NS Transformation Analysis (`ns_transformation_experiment.py`) 🆕
**Status**: New

Deep analysis of Newton-Schulz:
- Angular change between G and NS(G)
- Information loss: ||G - NS(G)||_F / ||G||_F
- Effect of varying NS steps (1, 2, 3, 5, 10)

### 6. Per-Component Gradient Analysis (`component_gradient_experiment.py`) 🆕
**Status**: New

Which layers benefit from which optimizers:
- Input/embedding layers vs hidden vs output
- NS benefit ratio per component
- Optimizer recommendations

### 7. Modular LR Scaling (`modular_lr_scaling_experiment.py`) 🆕
**Status**: New

Layer-wise learning rate budgeting:
- Depth-scaled LR strategies
- Gradient norm-aware scaling
- Adaptive LR per layer

---

## Next Steps
- [ ] Run understanding experiments
- [ ] Scale modular approach to larger models
- [ ] Investigate adaptive constraint selection during training
- [ ] Test with low-precision (fp16/bf16) training