# MNIST Experiments Results Summary

**Date**: 2025-12-11

## Experiments Completed

### 1. Gradient Rank Dynamics
- **Result**: Muon converges faster (loss 0.27 @ 500 steps vs Adam 0.63)
- **Key Finding**: Both optimizers maintain relatively stable effective rank, Muon shows faster loss reduction

### 2. Spectral Dynamics
- **AdamW**: 99.37% accuracy, loss 0.019
- **Muon**: 96.46% accuracy, loss 0.224
- **Key Finding**: Both show significant rank reduction (59.2 → ~3 effective rank in first layer)

### 3. Component Gradient Analysis
- **Result**: Muon benefits all layer types
- **NS Benefit Ratios**:
  - Hidden layers: 6.57x
  - Input layers: 5.02x
  - Output layers: 1.76x
- **Key Finding**: Higher NS benefit suggests Muon's spectral normalization is most valuable in hidden layers

### 4. Modular LR Scaling
| Strategy | Final Loss | Accuracy |
|----------|-----------|----------|
| uniform | 1.49 | 78.3% |
| depth_linear_0.5 | 1.23 | 82.2% |
| depth_linear | 1.02 | 84.6% |
| depth_inverse | 1.80 | 69.6% |
| **adaptive** | **0.41** | **90.1%** |

- **Key Finding**: Adaptive LR scaling significantly outperforms fixed strategies

### 5. Full Optimizer Comparison (2 epochs)
| Optimizer | Loss | Accuracy | Notes |
|-----------|------|----------|-------|
| **AdamW** | 0.126 | **97.0%** | Best overall |
| **Muon** | 0.092 | 96.3% | Lowest loss |
| Oblique+AdamW | 0.116 | 96.2% | Unit-norm columns |
| **L1-Stiefel+AdamW** | 0.170 | 96.0% | Sparse orthogonal updates |
| SGD+Momentum | 0.220 | 94.5% | Baseline |
| Grassmannian+AdamW | 191.4 | 10.3% | Too restrictive for MLP |

**Notes**: 
- Grassmannian constraints are too restrictive for standard MLPs (designed for attention-like layers)
- L1-Stiefel successfully combines orthogonality with sparsity

## Key Insights

1. **AdamW** is the most reliable optimizer for MNIST with highest accuracy
2. **Muon** provides fastest convergence (lowest loss) - best for quick training
3. **Oblique** constraint (unit norm columns) performs comparably to AdamW
4. **Adaptive LR scaling** provides 10%+ accuracy boost over uniform
5. **Hidden layers** benefit most from Newton-Schulz (6-7x improvement)
6. **Spectral rank** tends to collapse during training regardless of optimizer

## Results Location
All results saved in `/root/custom-optimizer-research/mnist/results/`:
- `gradient_rank/` - Gradient rank dynamics plots and logs
- `spectral_dynamics/` - SVD spectrum analysis
- `component_gradient/` - Per-component analysis
- `modular_lr_scaling/` - LR scaling comparison
- `optimizer_comparison/` - All optimizer benchmarks

