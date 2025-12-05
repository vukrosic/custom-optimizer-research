# Modular Manifolds for LLM Training: Using Different Optimizers for Different Layers

## Abstract

Standard practice in training large language models uses a single optimizer (typically AdamW) for all parameters. This paper explores a different approach inspired by the **Modular Manifolds** framework: using different optimizers and manifold constraints for different components of the network based on their mathematical properties.

We implement and compare three approaches:
1. **Muon optimizer** - Spectral normalization of gradient updates
2. **Stiefel manifold constraint** - Constraining weight matrices to have orthonormal columns (W^T W = I)
3. **Manifold Muon** - Combining both approaches

Our hypothesis is that weight matrices (attention projections, FFN layers) benefit from spectral-norm-aware optimization, while embeddings and normalization parameters are fine with standard AdamW.

---

## 1. Introduction

### The Problem with Uniform Optimization

When training transformers, we typically use AdamW for all parameters. But different parameters serve very different roles:

- **Embeddings**: Map discrete tokens to continuous vectors (lookup table)
- **Attention weights (QKV, Out)**: Transform input vectors, should preserve norms
- **FFN weights**: Non-linear feature transformation
- **Normalization**: Scale/shift activations

The key insight from the [Modular Manifolds](https://thinkingmachines.ai/blog/modular-manifolds/) article is that **weight matrices act as linear operators** on vectors. A matrix W transforms input x into output y = Wx. The **singular values** of W determine how much the matrix stretches or shrinks vectors along different directions.

### Why Singular Values Matter

For stable training, we want weight matrices to:
1. Not explode (singular values → ∞)
2. Not collapse (singular values → 0)
3. Have well-conditioned behavior (σ_max / σ_min not too large)

This motivates two approaches:

1. **Muon optimizer**: Normalize gradient updates to have unit spectral norm
2. **Stiefel manifold**: Constrain weight matrices so ALL singular values = 1

---

## 2. Background

### 2.1 The Muon Optimizer

Muon applies **Newton-Schulz iteration** to orthogonalize gradient updates:

```python
def newton_schulz(G, steps=5):
    """Orthogonalize gradient matrix G."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / G.norm()
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    return X
```

This computes the **matrix sign function**, which snaps all singular values to 1. The result is that gradient updates have unit spectral norm, meaning no single direction is updated disproportionately.

### 2.2 The Stiefel Manifold

The **Stiefel manifold** St(m,n) is the set of matrices with orthonormal columns:

$$\text{St}(m,n) = \{W \in \mathbb{R}^{m \times n} \mid W^T W = I_n\}$$

For a matrix on this manifold, ALL singular values equal 1. This means:
- The matrix neither stretches nor shrinks vectors
- The condition number is exactly 1
- The matrix acts as a pure rotation/reflection

### 2.3 Manifold Optimization

To optimize on a manifold:
1. Compute gradient in ambient space (standard backprop)
2. Project gradient to tangent space (for Stiefel: remove component that would break W^T W = I)
3. Take step in tangent direction
4. **Retract** back to manifold (project updated weights back to constraint set)

For Stiefel, the retraction can use Newton-Schulz (same as Muon) or QR decomposition.

---

## 3. Our Approach: Modular Optimization

We classify transformer parameters into groups and assign different optimization strategies:

| Parameter Group | Role | Optimizer | Manifold |
|-----------------|------|-----------|----------|
| **Embeddings** | Token → Vector | AdamW | None |
| **Attention (QKV, Out)** | Vector transformation | Muon or AdamW | Optional: Stiefel |
| **FFN (w1, w2, w3)** | Nonlinear mapping | Muon or AdamW | Optional: Stiefel |
| **Norms** | Scale activations | AdamW | None |

### Key Design Decisions

1. **Embeddings stay unconstrained**: They're a lookup table, not a linear operator
2. **Norm parameters stay unconstrained**: They're 1D vectors, not matrices
3. **Weight matrices can use Muon and/or Stiefel**: These are the "vector multipliers"

---

## 4. Experiments

We define 9 experiment configurations:

### Part 1: Optimizer Comparison (No Manifold Constraints)

| Experiment | Attention | FFN | Description |
|------------|-----------|-----|-------------|
| `baseline` | AdamW | AdamW | Standard training |
| `muon_all` | Muon | Muon | Muon for all matrices |
| `muon_attention` | Muon | AdamW | Muon for attention only |
| `muon_ffn` | AdamW | Muon | Muon for FFN only |

### Part 2: Manifold Constraints

| Experiment | Attention | FFN | Description |
|------------|-----------|-----|-------------|
| `stiefel_all` | Stiefel | Stiefel | Constrain all matrices to W^T W = I |
| `stiefel_attention` | Stiefel | AdamW | Stiefel for attention only |
| `stiefel_ffn` | AdamW | Stiefel | Stiefel for FFN only |
| `spectral_all` | Spectral | Spectral | Constrain σ_max = 1 only |

### Part 3: Combined Approach

| Experiment | Description |
|------------|-------------|
| `manifold_muon` | Muon optimizer + Stiefel constraint |

The `manifold_muon` experiment is the full implementation of "Manifold Muon" from the article: gradient updates are orthogonalized (Muon), AND weights are constrained to the Stiefel manifold.

---

## 5. Implementation

### 5.1 Stiefel Projection via Newton-Schulz

We use the same Newton-Schulz iteration for both Muon and Stiefel retraction:

```python
def stiefel_project_newton_schulz(W, steps=5):
    """Project W onto Stiefel manifold."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = W / W.norm()
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    return X
```

### 5.2 Stiefel Optimizer Wrapper

```python
class StiefelOptimizer:
    """Wraps any optimizer with Stiefel manifold constraint."""
    
    def __init__(self, params, base_optimizer, ns_steps=5):
        self.base_optimizer = base_optimizer
        self.ns_steps = ns_steps
        
        # Initialize weights on manifold
        for p in params:
            if p.ndim >= 2:
                p.data = stiefel_project_newton_schulz(p.data)
    
    def step(self):
        # Take base optimizer step
        self.base_optimizer.step()
        
        # Retract back to manifold
        for p in self.matrix_params:
            p.data = stiefel_project_newton_schulz(p.data)
```

### 5.3 Parameter Classification

```python
def classify_parameter(name):
    if 'embed' in name or 'lm_head' in name:
        return 'embedding'
    if 'attention' in name or 'qkv' in name:
        return 'attention'
    if 'feed_forward' in name or 'w1' in name or 'w2' in name:
        return 'ffn'
    if 'norm' in name:
        return 'norm'
    return 'other'
```

---

## 6. Hypotheses and Expected Results

### H1: Muon improves training stability
Spectral normalization of updates should prevent any single weight direction from dominating, leading to more stable training.

### H2: Stiefel constraint prevents weight explosion
By constraining all singular values to 1, weights cannot grow unboundedly. This eliminates need for weight decay.

### H3: Attention benefits more than FFN from Muon/Stiefel
Attention weights directly transform the input representation and should maintain good conditioning. FFN weights include the SwiGLU nonlinearity which may disrupt the benefit.

### H4: Manifold Muon provides best of both worlds
Combining orthogonalized updates (Muon) with weight constraints (Stiefel) should give the most stable training.

### H5: Weight decay becomes unnecessary with Stiefel
Since weights are constrained, explicit regularization is redundant.

---

## 7. Metrics to Track

For each experiment, we will track:

1. **Training loss** - Optimization effectiveness
2. **Validation loss / Perplexity** - Generalization
3. **Singular value distribution** per layer - Weight conditioning
4. **Weight Frobenius norm** per layer - Weight growth
5. **Gradient norm** per layer - Gradient stability
6. **Training speed** (tokens/sec) - Computational overhead

---

## 8. Code Structure

```
experiments/
├── __init__.py
├── experiment_config.py    # Experiment definitions
├── manifold_constraints.py # Stiefel & Spectral optimizers
├── modular_optimizer.py    # Per-layer optimizer assignment
├── run_experiments.py      # Training script
└── analyze_results.py      # Result visualization
```

### Running Experiments

```bash
# Single experiment
python experiments/run_experiments.py --exp baseline --max_steps 2000

# All experiments
python experiments/run_experiments.py --all

# Analyze results
python experiments/analyze_results.py --results_dir experiment_results
```

---

## 9. Related Work

- **Muon optimizer**: Kosson et al. (2024), "Muon: Momentum Orthogonalized by Newton-Schulz"
- **Modular Manifolds**: Bernstein (2025), Thinking Machines Lab
- **Polar Express**: Fast Newton-Schulz computation on GPUs
- **Weight normalization**: Salimans & Kingma (2016)
- **Spectral normalization**: Miyato et al. (2018)

---

## 10. Conclusion

This paper presents a framework for **modular optimization** of large language models, where different parameter groups use different optimizers and manifold constraints based on their mathematical role in the network.

The key contributions are:
1. A classification scheme for transformer parameters
2. Implementation of Stiefel and spectral norm constraints
3. A modular optimizer that combines different strategies
4. Experimental framework to compare approaches

We hypothesize that applying spectral-aware optimization (Muon) and manifold constraints (Stiefel) to weight matrices while leaving embeddings and norms unconstrained will improve training stability without sacrificing expressivity.

---

## Appendix: Theoretical Background

### A.1 Why Newton-Schulz for Stiefel?

The Newton-Schulz iteration computes the **polar decomposition** of a matrix:

$$W = UP$$

where U has orthonormal columns and P is positive semi-definite. The matrix U is the nearest orthonormal matrix to W, making it the Stiefel projection.

### A.2 Condition Number and Expressivity

A concern with Stiefel constraints is that they limit expressivity by forcing σ = 1 for all singular values. However:

1. The network has multiple layers, so overall mapping is still expressive
2. Normalization layers (RMSNorm) provide scaling
3. The constraint may act as implicit regularization

### A.3 Learning Rate Considerations

With Stiefel constraints, learning rate semantics change:
- Updates are in the tangent space, not ambient space
- The step size directly corresponds to geodesic distance on the manifold
- Lower learning rates may be needed to stay on manifold accurately
