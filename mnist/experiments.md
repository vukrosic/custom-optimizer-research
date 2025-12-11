# Experiment Documentation

This document describes all experiments in the custom-optimizers-research project.

---

## Research Goal

Understand how different optimization algorithms and geometric constraints affect training dynamics, gradient structure, and convergence behavior.

---

## Existing Experiments

### 1. Gradient Rank Experiment
**File**: `mnist/experiments/gradient_rank_experiment.py`  
**Goal**: Track how effective rank changes during training with Adam vs Muon

Measures the "information capacity" of gradients using Shannon entropy-based effective rank. The hypothesis is that orthonormal updates (Muon) preserve gradient rank better than raw gradient descent (Adam).

**Key Metrics**:
- Effective rank before/after Newton-Schulz
- Top-k singular value concentration
- Loss curves

---

### 2. Spectral Dynamics Experiment
**File**: `mnist/experiments/spectral_dynamics_experiment.py`  
**Goal**: Full SVD spectrum tracking throughout training

Goes beyond effective rank to track the complete singular value distribution. Helps understand:
- Spectral collapse patterns
- Condition number dynamics  
- Early vs late training differences

**Key Metrics**:
- Complete singular value spectrum
- Condition number over time
- Top-1, Top-5, Top-10 concentration ratios

---

### 3. Newton-Schulz Transformation Experiment
**File**: `mnist/experiments/ns_transformation_experiment.py`  
**Goal**: Analyze how Newton-Schulz transforms gradients

Deep dive into the NS iteration itself:
- Angular change between G and NS(G)
- Information loss: ||G - NS(G)||_F / ||G||_F  
- Effect of varying NS steps (1, 2, 3, 5, 10)

---

### 4. Component Gradient Experiment
**File**: `mnist/experiments/component_gradient_experiment.py`  
**Goal**: Which network layers benefit from which optimizers

Based on the Modular Manifolds hypothesis: different components have different optimization needs:
- **Input layers** (embeddings): May benefit from Sphere/Oblique
- **Hidden layers**: May benefit from Muon/Stiefel
- **Output layers**: Standard AdamW often sufficient

**Key Outputs**: Per-component optimizer recommendations

---

### 5. Modular LR Scaling Experiment
**File**: `mnist/experiments/modular_lr_scaling_experiment.py`  
**Goal**: Test per-layer learning rate strategies

Based on the modular-manifolds concept of "scalar coefficients budgeting learning rates across layers":
- Uniform LR (baseline)
- Depth-scaled LR (smaller for deeper)
- Gradient-norm-aware scaling
- Inverse-depth scaling

---

### 6. MNIST NS Experiment
**File**: `mnist/experiments/mnist_ns_experiment.py`  
**Goal**: Compare AdamW vs Muon with different Newton-Schulz iterations (3 vs 5)

Systematic comparison of:
- AdamW baseline
- Muon with 3 NS iterations
- Muon with 5 NS iterations

Tracks average effective rank and accuracy.

---

## Planned Experiments

### 7. Optimizer Comparison
**File**: `mnist/experiments/optimizer_comparison.py` (to create)  
**Goal**: Systematically compare all new optimizers on MNIST

Tests:
- AdamW, Muon (baseline)
- Oblique (for input layer)
- Grassmannian (subspace optimization)
- Block-Stiefel (if using attention)
- L1-Stiefel (sparse updates)

**Metrics**: Loss, accuracy, effective rank, convergence speed, wall-clock time

---

### 8. Manifold Constraint Ablation
**File**: `mnist/experiments/manifold_ablation.py` (to create)  
**Goal**: Ablate each constraint individually

Configurations:
- Baseline: AdamW only
- +Sphere on input layer
- +Stiefel on hidden layers
- +Both

---

### 9. Convergence Stability Test
**File**: `mnist/experiments/convergence_stability.py` (to create)  
**Goal**: Long-run numerical stability

Run 100+ epochs and track:
- Gradient norm explosion/vanishing
- Weight norm drift
- NaN/Inf occurrences
- Condition number stability

---

## Running Experiments

```bash
# From project root
cd /Users/vukrosic/AI\ Science\ Projects/custom-optimizers-research

# Run a single MNIST experiment
python -m mnist.experiments.gradient_rank_experiment --epochs 5

# Run with both optimizers
python -m mnist.experiments.spectral_dynamics_experiment --optimizer both
```
