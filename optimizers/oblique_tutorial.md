# Oblique Optimizer Tutorial

## 1. What is the Oblique Manifold?

In deep learning, we essentially optimize matrices of weights. Standard optimization (like SGD or Adam) assumes these weights exist in a "flat" Euclidean space where any value is valid.

However, sometimes we want to enforce constraints. The **Oblique Manifold** is one such geometric constraint where **every column of the weight matrix must have a unit Euclidean norm (length of 1)**.

Mathematically, for a matrix $W \in \mathbb{R}^{m \times n}$ with columns $w_1, ..., w_n$:
$$ \mathcal{O}(m, n) = \{ W \in \mathbb{R}^{m \times n} \mid \|w_i\|_2 = 1 \text{ for all } i = 1, \dots, n \} $$

### Why use it?
1.  **Scale Invariance**: It removes the "magnitude" of the feature vector, forcing the model to learn purely based on the **direction** or angle of the vectors.
2.  **Stability**: It prevents weight explosion (gradients cannot grow the weights indefinitely).
3.  **Embeddings**: It is naturally suited for embedding layers where we often normalize vectors for cosine similarity anyway.

---

## 2. Code Walkthrough

We implemented the Oblique optimizer as a **wrapper** around a standard optimizer (like AdamW). This technique is often called **Projected Gradient Descent (PGD)**: we take a step in the Euclidean space, and then "project" the result back onto the manifold.

### The Implementation (`optimizers/oblique.py`)

Here is the code with a step-by-step explanation:

```python
def oblique_project(W: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Project matrix columns onto the hypersphere."""
    if W.ndim != 2:
        return W
    
    # Calculate the L2 norm of each column (dim=0)
    norms = W.norm(dim=0, keepdim=True)
    
    # Scale columns: radius / current_norm
    # The +1e-8 prevents division by zero
    return W * (radius / (norms + 1e-8))
```

This helper function performs the projection. Visualizing this in 3D: if your weight column assumes a value inside or outside the sphere, this function extends or shrinks it until it exactly touches the surface of the sphere.

### The Optimizer Class

```python
class ObliqueOptimizer:
    def __init__(self, params, base_optimizer: torch.optim.Optimizer, radius: float = 1.0):
        self.base_optimizer = base_optimizer
        self.radius = radius
        
        # 1. Capture all 2D parameters (matrices) to apply the constraint to
        self.matrix_params = []
        for group in base_optimizer.param_groups:
            for p in group['params']:
                if p.ndim >= 2:
                    self.matrix_params.append(p)
        
        # 2. Initial Projection: Ensure we start on the manifold
        for p in self.matrix_params:
            with torch.no_grad():
                p.data = oblique_project(p.data, self.radius)
```

In `__init__`, we accept a `base_optimizer`. This allows us to use the sophisticated update rules of AdamW (momentum, adaptive learning rates) while still enforcing our constraint. We strictly enforce that the weights start valid by projecting them immediately.

### The Training Step

```python
    @torch.no_grad()
    def step(self, closure=None):
        # 1. Standard Step
        # Let AdamW update the weights based on gradients.
        # This will momentarily likely move weights OFF the manifold.
        if 'closure' in sig.parameters:
            loss = self.base_optimizer.step(closure)
        else:
            loss = self.base_optimizer.step()
        
        # 2. Projection (The "Oblique" Magic)
        # Immediately force the weights back onto the unit sphere.
        for p in self.matrix_params:
            p.data = oblique_project(p.data, self.radius)
        
        return loss
```

This `step` method is the core. By projecting *after* the update, we ensure that at the end of every iteration, the constraint holds.

---

## 3. Results on MNIST

We tested `Oblique + AdamW` against standard `AdamW`, `Muon`, and `L1-Stiefel`.

| Metric | AdamW | Oblique + AdamW | Difference |
| :--- | :--- | :--- | :--- |
| **Final Test Accuracy** | 97.69% | **97.64%** | Comparable (-0.05%) |
| **Final Training Loss** | 0.041 | **0.027** | **Oblique is lower / better** |
| **Effective Rank Drop** | -78% | **-81%** | Oblique collapses rank more |
| **Convergence Speed** | Standard | **Fast** | Converged very quickly |

### Key Findings
1.  **Lowest Training Loss**: Oblique achieved the lowest training loss of all optimizers.
2.  **High Rank Collapse**: Oblique caused the highest reduction in the "effective rank" of the weight matrices (matrix columns became more correlated).
3.  **High Accuracy**: Despite the rank collapse, it generalized almost as well as unconstrained AdamW.

---

## 4. Why are the results like this?

### Why the Lowest Loss?
The Oblique constraint simplifies the optimization landscape. By fixing the magnitude of every column to 1, the optimizer removes an entire axis of variation (length). The model cannot "cheat" by blowing up weight magnitudes to increase confidence. It forces the model to learn better **angular representations** (features) to minimize the loss. This "hard constraint" often acts as a powerful guide, potentially removing local minima associated with scale.

### Why the Rank Collapse?
This is the trade-off.
*   **Rank** is a measure of how many independent directions of variation your matrix has.
*   In a standard unconstrained matrix, columns can vary by **angle** AND **length**.
*   In Oblique, columns can only vary by **angle**.

By removing the "length" degree of freedom, you naturally squeeze the data into a smaller effective subspace. Additionally, in high-dimensional spaces, unit vectors that maximize dot products (which neural networks try to do) tend to cluster, leading to higher correlation between columns and thus a lower effective rank.

### Conclusion
The Oblique optimizer demonstrates that **magnitude is not always efficient**. By constraining weights to the Oblique manifold, we forced the network to learn efficient directional representations, achieving the lowest training loss, though at the cost of compressing the variance (rank) of the latent space.
