# Information Flow & Gradient Quality Hypothesis

**Research Goal**: Understand *why* orthonormalizing weight updates (as Muon does) improves learning, rather than just *that* it does.

---

## Core Hypothesis

> **Orthonormal updates preserve the "information capacity" of gradients better than raw gradients, preventing collapse into low-rank subspaces.**

When training neural networks, gradients can degenerate in several ways:
1. **Rank collapse**: Gradients may become low-rank over time, meaning updates only affect a few directions in weight space
2. **Spectral imbalance**: Some singular values dominate, causing learning to be fast in some directions and slow in others
3. **Noise amplification**: In high-dimensional spaces, noise can overwhelm signal in certain gradient dimensions

The Muon optimizer projects gradients onto the Stiefel manifold (orthonormal matrices) via Newton-Schulz iterations. This transformation:
- Forces all singular values to 1
- Maintains full rank in the update matrix
- Equalizes "learning intensity" across all directions

**If this hypothesis is correct, we should observe**:
1. Raw gradients becoming increasingly low-rank during training (especially late-stage)
2. Orthonormalized updates maintaining higher effective rank throughout training
3. Correlation between gradient rank collapse and learning stagnation

---

## Theoretical Background

### The Newton-Schulz Transformation

The Muon optimizer uses Newton-Schulz iterations to approximate the "zeroth power" of a matrix:

```
X_{k+1} = X_k * (1.5*I - 0.5*X_k^T*X_k)  (simplified)
```

This converges to the closest orthonormal matrix to G, which is `U @ V^T` from the SVD `G = U @ S @ V^T`. This effectively:
- Replaces all singular values with 1
- Removes spectral information but preserves directional information

### Effective Rank

The **effective rank** quantifies how "spread out" the singular values are:

```
effective_rank(A) = exp(H(σ)) 
where H(σ) = -Σ p_i * log(p_i)
and p_i = σ_i / Σσ_j
```

- For a matrix with all equal singular values: effective_rank ≈ min(m, n)
- For a rank-1 matrix: effective_rank = 1
- This is a continuous measure that captures "how many dimensions matter"

### Gradient Signal-to-Noise Ratio

We define the gradient SNR as:

```
SNR = ||E[g]||_F / std(g)
```

Where the expectation is over minibatches. High SNR indicates consistent gradient direction.

---

## Experiment Design

### Experiment 1: Gradient Rank Dynamics Over Training

**Objective**: Track how the effective rank of gradients evolves during training for both Muon and Adam.

**Metrics to collect** (per layer, per training step):
- [ ] Full singular value spectrum of gradient: `σ = svd(grad)[1]`
- [ ] Effective rank: `exp(-sum(p * log(p)))` where `p = σ/sum(σ)`
- [ ] Top-k singular value ratio: `sum(σ[:k]) / sum(σ)` for k=1,5,10
- [ ] Frobenius norm of gradient: `||grad||_F`
- [ ] Gradient magnitude per layer

**Procedure**:
```python
# Pseudocode for gradient rank tracking
for step in training_steps:
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad.ndim == 2:  # Only 2D weight matrices
            grad = param.grad.detach().float()
            
            # Compute singular values
            U, S, V = torch.linalg.svd(grad, full_matrices=False)
            
            # Normalize for entropy calculation
            p = S / S.sum()
            entropy = -(p * torch.log(p + 1e-10)).sum()
            effective_rank = torch.exp(entropy)
            
            # Log metrics
            log({
                f"{name}/effective_rank": effective_rank,
                f"{name}/top1_ratio": S[0] / S.sum(),
                f"{name}/condition_number": S[0] / S[-1],
                f"{name}/frobenius_norm": grad.norm(),
            })
    
    optimizer.step()
```

**Hypothesis predictions**:
- Adam: Effective rank decreases over training (especially for deeper layers)
- Muon: Effective rank remains stable (since updates are always full-rank orthonormal)

---

### Experiment 2: Before/After Newton-Schulz Comparison

**Objective**: Compare gradient properties before and after the Newton-Schulz transformation at various training stages.

**Metrics**:
- [ ] Singular value spectrum before NS (the momentum-adjusted gradient)
- [ ] Singular value spectrum after NS (should be all 1s, but measure empirically)
- [ ] Angle between original gradient and NS-transformed gradient
- [ ] Information loss: `||G - G_ns||_F`

**Procedure**:
```python
def analyze_newton_schulz_effect(G, steps=5):
    """Analyze what Newton-Schulz does to a gradient matrix."""
    
    # Before transformation
    U_pre, S_pre, V_pre = torch.linalg.svd(G, full_matrices=False)
    
    # After transformation
    G_ns = zeropower_via_newtonschulz5(G, steps=steps)
    U_post, S_post, V_post = torch.linalg.svd(G_ns, full_matrices=False)
    
    # Metrics
    angle = torch.acos((G.flatten() @ G_ns.flatten()) / (G.norm() * G_ns.norm()))
    info_loss = (G - G_ns).norm() / G.norm()
    rank_preservation = effective_rank(G_ns) / effective_rank(G)
    
    return {
        "pre_singular_values": S_pre,
        "post_singular_values": S_post,  # Should be ~1.0
        "angle_degrees": angle * 180 / π,
        "relative_info_loss": info_loss,
        "rank_change_ratio": rank_preservation,
    }
```

**Key questions**:
1. How much does the gradient direction change after orthonormalization?
2. Is there a "sweet spot" of gradient rank where orthonormalization helps most?
3. Do near-orthogonal gradients get transformed less?

---

### Experiment 3: Controlled Rank Injection

**Objective**: Artificially manipulate gradient rank to test causality.

**Procedure**:
1. Train with Adam, but periodically project gradients to different ranks
2. Compare learning curves for:
   - Full-rank gradients (no projection)
   - Low-rank gradients (keep only top-k singular values)
   - Orthonormalized gradients (Muon-style)
   - Re-inflated gradients (scale singular values to be equal but not 1)

**Implementation**:
```python
class RankManipulatingOptimizer(torch.optim.Adam):
    def __init__(self, *args, rank_mode="full", keep_rank=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_mode = rank_mode
        self.keep_rank = keep_rank
    
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.grad.ndim != 2:
                    continue
                
                grad = p.grad
                if self.rank_mode == "low_rank":
                    # Keep only top-k singular values
                    U, S, V = torch.linalg.svd(grad, full_matrices=False)
                    S[self.keep_rank:] = 0
                    p.grad = U @ torch.diag(S) @ V
                    
                elif self.rank_mode == "orthonormal":
                    # Muon-style projection
                    p.grad = zeropower_via_newtonschulz5(grad)
                    
                elif self.rank_mode == "equalized":
                    # Make all singular values equal (but preserve total energy)
                    U, S, V = torch.linalg.svd(grad, full_matrices=False)
                    S_eq = torch.ones_like(S) * S.mean()
                    p.grad = U @ torch.diag(S_eq) @ V
        
        super().step()
```

**Hypothesis predictions**:
- Low-rank projection will hurt learning
- Orthonormalization will help
- Equalization will help but less than orthonormalization

---

### Experiment 4: Per-Layer Rank Analysis

**Objective**: Understand which layers benefit most from orthonormalization.

**Focus areas**:
- [ ] Embedding layers
- [ ] QKV projections in attention
- [ ] Output projections
- [ ] FFN layers (W1, W2, W3)
- [ ] LM head

**Metrics per layer**:
- Gradient effective rank vs layer depth
- Correlation between layer rank and layer's contribution to loss

**Hypothesis**: Earlier layers may have higher-rank gradients (more diverse signals), while later layers may have lower-rank gradients (more specialized).

---

### Experiment 5: Gradient Coherence Across Minibatches

**Objective**: Test if orthonormalization improves gradient signal-to-noise ratio.

**Procedure**:
```python
def measure_gradient_snr(model, dataloader, num_batches=100):
    """Measure gradient signal-to-noise ratio across minibatches."""
    gradients = {name: [] for name, p in model.named_parameters() if p.requires_grad}
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        model.zero_grad()
        loss = model(batch).loss
        loss.backward()
        
        for name, p in model.named_parameters():
            if p.grad is not None:
                gradients[name].append(p.grad.clone())
    
    snr_results = {}
    for name, grads in gradients.items():
        grads = torch.stack(grads)  # [num_batches, *grad_shape]
        mean_grad = grads.mean(dim=0)
        std_grad = grads.std(dim=0)
        
        snr = mean_grad.norm() / (std_grad.norm() + 1e-10)
        snr_results[name] = snr.item()
    
    return snr_results
```

**Extended analysis**: Compare SNR of raw gradients vs orthonormalized gradients.

---

## Implementation Plan

### Phase 1: Instrumentation (Week 1)
1. Create a `GradientAnalyzer` class that hooks into training
2. Implement efficient SVD computation for large matrices (possibly sampled)
3. Set up logging infrastructure (wandb or tensorboard)

### Phase 2: Baseline Experiments (Week 2)  
1. Run Experiment 1 with Adam baseline
2. Run Experiment 1 with Muon
3. Compare gradient rank dynamics

### Phase 3: Mechanistic Analysis (Week 3)
1. Run Experiment 2: Newton-Schulz analysis
2. Run Experiment 3: Controlled rank manipulation
3. Test causal hypotheses

### Phase 4: Deep Analysis (Week 4)
1. Run Experiment 4: Per-layer analysis
2. Run Experiment 5: Cross-batch coherence
3. Synthesize findings

---

## Expected Outcomes & Theory Development

### If hypothesis is SUPPORTED:
- We would conclude that **gradient rank preservation is a key mechanism** of Muon's effectiveness
- This suggests designing optimizers that explicitly maintain gradient diversity
- Could lead to lightweight alternatives (cheap rank-preserving operations)

### If hypothesis is REFUTED:
- Orthonormalization helps for reasons OTHER than rank preservation
- Alternative hypotheses to explore:
  - **Curvature normalization**: Orthonormal updates may implicitly handle loss landscape curvature
  - **Implicit regularization**: Forcing orthonormal updates may regularize toward better solutions
  - **Optimization geometry**: The Stiefel manifold may have favorable optimization properties

### Potential theoretical contributions:
1. **A measure of "gradient health"** that predicts optimizer-dependent learning
2. **Understanding of spectral collapse** in neural network training
3. **Principled guidelines** for when Muon-style optimizers are most beneficial

---

## Files to Create

| File | Purpose |
|------|---------|
| `experiments/gradient_analyzer.py` | Core instrumentation for gradient metrics |
| `experiments/run_gradient_experiment.py` | Main experiment runner |
| `experiments/rank_manipulation.py` | Implements controlled rank experiments |
| `experiments/visualize_gradients.py` | Plotting and visualization utilities |

---

## Success Criteria

The experiment is successful if we can answer:

1. ✅ Does gradient effective rank change during training? How?
2. ✅ Is there a difference in rank dynamics between Adam and Muon?
3. ✅ Does artificially manipulating gradient rank affect learning?
4. ✅ Which aspects of orthonormalization matter most (direction preservation? rank inflation? spectral equalization?)

---

## References

- Muon optimizer: Newton-Schulz orthogonalization for momentum
- Matrix spectral analysis for deep learning
- Effective rank / intrinsic dimensionality measures
- Natural gradient and Fisher information connections
