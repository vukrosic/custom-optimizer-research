# Oblique vs Stiefel Matrix Dynamics Experiment

## Overview

This experiment compares three optimization strategies to understand how different geometric constraints affect learning dynamics and weight matrix properties:

1. **Muon (Baseline)**: Uses Muon optimizer for 2D weight matrices with no manifold constraints
2. **Oblique**: Unit-norm columns (each column has L2 norm = 1, but columns can be correlated)
3. **Stiefel**: Orthonormal columns (each column has L2 norm = 1 AND columns are orthogonal: W^T W = I)

## Research Question

How do different geometric constraints on weight matrices affect:
- Learning dynamics and convergence
- Column correlation (orthogonality)
- Weight sparsity
- Final model performance

## Experimental Setup

### Model
- **Architecture**: 3-layer MLP for MNIST classification
- **Hidden sizes**: [256, 128]
- **Dataset**: MNIST (28×28 grayscale digits)

### Optimizers
All three configurations use:
- **Muon optimizer** for 2D weight matrices (lr=0.02, momentum=0.95)
- **AdamW optimizer** for other parameters (bias, etc., lr=1e-3)

The key difference is the post-processing after each optimization step:
- **Muon**: No post-processing (baseline)
- **Oblique**: Projects weights to unit-norm columns (normalizes each column)
- **Stiefel**: Projects weights to orthonormal columns using Newton-Schulz iteration

### Fair Initialization
To ensure fair comparison, we normalize initialization scales so all three start from similar output scales. This isolates the effect of learning dynamics rather than initialization artifacts.

## Key Findings

### 1. Off-Diagonal Correlation (Column Orthogonality)

**Muon (Baseline)**:
- Starts at ~0.066, increases to ~0.35 over training
- Columns become increasingly correlated (no constraint to prevent this)

**Oblique**:
- Maintains stable correlation at ~0.05 throughout training
- Normalization alone prevents correlation from growing

**Stiefel**:
- Maintains very low correlation at ~0.01 throughout training
- Enforces strict orthogonality (W^T W = I)

### 2. Weight Sparsity

**Muon**: ~15% → 9.7% (decreases over time)

**Oblique**: ~11.7% → 13.9% (slight increase, stable)

**Stiefel**: ~29.8% → 39.0% (high and increasing)

**Key Insight**: Stiefel's orthonormalization process naturally induces sparsity as a side effect. The Newton-Schulz iteration that enforces orthogonality redistributes weight magnitudes, pushing some weights toward zero.

### 3. Learning Dynamics

**Epoch 1 Loss**:
- Muon: 0.2565 (best start)
- Oblique: 0.2910
- Stiefel: 0.2775

**Final Loss (Epoch 5)**:
- Muon: 0.1332
- Oblique: 0.1325 (best)
- Stiefel: 0.1505

**Observations**:
- **Muon**: Starts best but degrades - correlation increases, loss rises slightly
- **Oblique**: Stable learning, best final performance
- **Stiefel**: Slightly worse final loss, but maintains strict geometric properties

### 4. Orthogonality Error

Stiefel maintains near-perfect orthogonality (||W^T W - I||_F ≈ 0), while Oblique and Muon show increasing orthogonality error over time.

## Technical Details

### Oblique Projection
```python
# Simple column normalization
norms = W.norm(dim=0, keepdim=True)
W_normalized = W * (radius / (norms + 1e-8))
```

### Stiefel Projection (Newton-Schulz)
```python
# Iterative orthonormalization
for _ in range(steps):
    A = X @ X.T
    B = b * A + c * A @ A
    X = a * X + B @ X
```

The Newton-Schulz iteration is more computationally expensive but enforces strict orthogonality.

## Why Stiefel Creates Sparsity

The orthonormalization process requires columns to be uncorrelated. When columns are similar, the Newton-Schulz iteration pushes them apart to achieve orthogonality. This redistribution of weight magnitudes can drive some weights toward zero, creating sparsity as a side effect of enforcing orthogonality.

## Implications for Learning and Intelligence

1. **Geometric Constraints Matter**: The choice of manifold constraint significantly affects learning dynamics, not just final performance.

2. **Orthogonality vs. Performance Trade-off**: Stiefel maintains perfect orthogonality but at a slight cost to final loss. Oblique achieves better performance while maintaining reasonable correlation.

3. **Sparsity Emerges from Geometry**: High sparsity in Stiefel is not from explicit regularization (no L1), but from the geometric constraint itself.

4. **Stability**: Both Oblique and Stiefel maintain stable geometric properties throughout training, while the unconstrained baseline (Muon) shows degradation.

## Files

- `oblique_vs_stiefel.py`: Main experiment script
- `oblique_vs_stiefel_matrix_dynamics.png`: Visualization of results
- `README.md`: This report

## Running the Experiment

```bash
cd mnist/experiments/oblique_vs_stiefel_experiment
python oblique_vs_stiefel.py
```

## Future Directions

1. Investigate why Stiefel's sparsity increases over time
2. Explore hybrid approaches (e.g., soft orthogonality constraints)
3. Test on larger models and datasets
4. Analyze the relationship between sparsity and generalization
5. Study the computational cost vs. benefit trade-offs


---

Why does orthonormality error rise so much with maseline muon (in the image), does it mean that neurons are pointing in the same direction, redundant? overfitting?