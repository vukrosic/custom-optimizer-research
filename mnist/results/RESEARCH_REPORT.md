# MNIST Optimizer Research Report

## Matrix Transformations and Singular Value Analysis

**Date**: December 2024  
**Focus**: How different optimizers transform weight matrices during training

---

## Executive Summary

This report analyzes 5 optimizers on MNIST, focusing on **how each optimizer transforms weight matrices** through training. We track singular value spectra, effective rank, condition number, and orthogonality to understand the geometric properties each optimizer induces.

| Optimizer | Accuracy | Loss | Key Characteristic |
|-----------|----------|------|-------------------|
| **AdamW** | **97.69%** | 0.041 | Steady rank reduction, best accuracy |
| **Oblique** | **97.64%** | **0.027** | Unit-norm columns, lowest loss |
| Muon | 96.96% | 0.082 | Spectral normalization, fast early convergence |
| L1-Stiefel | 96.95% | 0.115 | Sparse orthogonal updates |
| SGD | 96.85% | 0.102 | Baseline, slowest convergence |

---

## 1. Singular Value Analysis

The singular value spectrum reveals how each optimizer shapes the weight matrices.

![Singular Values](/mnist/results/analysis/singular_values.png)

### Key Observations

**fc1 (Input Layer - 784→256)**
- All optimizers show steep decay in singular values
- **Muon** produces the flattest spectrum (more uniform SVs)
- **AdamW/Oblique** show sharper top SV concentration

**fc2 (Hidden Layer - 256→256)**
- Square matrix allows cleaner SVD analysis
- **L1-Stiefel** maintains more uniform SVs due to orthogonality constraint
- **Oblique** shows unit-norm column effect

**fc3 (Output Layer - 256→10)**
- All optimizers converge to similar spectra
- Low-rank structure emerges (only 10 output classes)

---

## 2. Matrix Transformation Analysis

Tracking how matrix properties evolve during training:

![Matrix Analysis](/mnist/results/analysis/matrix_analysis.png)

### Effective Rank

Effective rank measures how many singular values contribute meaningfully:

$$\text{eff\_rank} = \exp\left(-\sum_i \sigma_i \log \sigma_i\right)$$

| Optimizer | Initial Rank | Final Rank | Change |
|-----------|--------------|------------|--------|
| AdamW | ~180 | ~40 | -78% |
| Muon | ~180 | ~50 | -72% |
| Oblique | ~180 | ~35 | -81% |
| L1-Stiefel | ~180 | ~45 | -75% |
| SGD | ~180 | ~60 | -67% |

**Insight**: All optimizers reduce effective rank, but **SGD preserves the most rank** while **Oblique collapses the most**.

### Condition Number

Condition number (σ_max/σ_min) indicates numerical stability:

- **L1-Stiefel**: Lowest condition number (best conditioned due to orthogonality)
- **AdamW/Oblique**: Similar conditioning
- **Muon**: Spectral normalization helps maintain conditioning

### Orthogonality Error

Measures deviation from orthogonal columns: ||W^T W - I||_F

- **L1-Stiefel**: Explicitly enforces orthogonality → lowest error
- **Oblique**: Unit-norm columns but allows correlation
- **Muon**: Newton-Schulz orthogonalizes gradients, not weights
- **AdamW/SGD**: No orthogonality constraint

---

## 3. Training Dynamics

![Training Curves](/mnist/results/analysis/training_curves.png)

### Convergence Speed

| Optimizer | Steps to 95% | Steps to 96% |
|-----------|--------------|--------------|
| **Muon** | ~100 | ~150 |
| AdamW | ~150 | ~300 |
| Oblique | ~150 | ~400 |
| L1-Stiefel | ~200 | ~500 |
| SGD | ~400 | ~700 |

**Muon converges fastest** due to spectral-normalized updates.

---

## 4. Gradient Analysis

![Gradient Analysis](/mnist/results/analysis/gradient_analysis.png)

### Gradient Effective Rank

The rank of gradients indicates information capacity:

- **Muon** maintains higher gradient rank due to Newton-Schulz orthogonalization
- **AdamW** gradient rank decreases as training progresses
- **L1-Stiefel** has lower gradient rank due to sparsification

---

## 5. Final Comparison

![Final Comparison](/mnist/results/analysis/final_comparison.png)

---

## 6. Optimizer Characteristics Summary

### AdamW
- **Best for**: General-purpose training
- **Matrix effect**: Moderate rank reduction
- **Trade-off**: Reliable but no geometric constraints

### Muon (Newton-Schulz)
- **Best for**: Fast initial convergence
- **Matrix effect**: Spectral normalization of gradients
- **Trade-off**: May plateau; gradients orthogonalized, not weights

### Oblique
- **Best for**: Embedding/normalization layers
- **Matrix effect**: Unit-norm columns enforced
- **Trade-off**: Highest rank collapse

### L1-Stiefel
- **Best for**: Sparse + orthogonal solutions
- **Matrix effect**: Maintains orthogonality, sparse updates
- **Trade-off**: Slower convergence

### SGD
- **Best for**: Baseline comparison
- **Matrix effect**: Preserves most rank
- **Trade-off**: Slowest convergence

---

## 7. Conclusions

1. **Rank Collapse is Universal**: All optimizers reduce effective rank by 67-81%
2. **Geometric Constraints Help**: Oblique/L1-Stiefel achieve competitive accuracy with structure
3. **Speed vs Structure Trade-off**: Muon converges fastest; L1-Stiefel is most structured
4. **Singular Value Spectra**: Muon produces flattest spectrum; others show sharper decay
5. **Orthogonality**: Only L1-Stiefel explicitly maintains orthogonal weights

---

## Files

All results in `mnist/results/analysis/`:
- `training_curves.png` - Loss/accuracy over time
- `matrix_analysis.png` - Rank, conditioning, orthogonality
- `singular_values.png` - SVD spectra per layer  
- `gradient_analysis.png` - Gradient statistics
- `final_comparison.png` - Summary bar charts
- `metrics.json` - Raw numerical data
