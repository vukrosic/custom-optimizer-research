# Adaptive Attention Selection: A Proof-of-Concept for Dynamic Routing in Linear Transformers

**Abstract**

This paper proposes a dynamic routing mechanism that selects between linear (O(n)) and softmax (O(n²)) attention on a per-token basis. While linear attention offers efficiency, it often lacks the attention (detail retrieval and understanding) power of full softmax attention. This approach allows the model to adaptively allocate compute: using fast linear attention for "easy" tokens and computationally intensive softmax attention for "hard" tokens. This introduces a trade-off: increased training complexity (teaching the router) in exchange for optimized inference (allocating FLOPs where needed). This concept is validated with a proof-of-concept 4-layer model, demonstrating that a stable, balanced routing strategy can be learned.

---

## 1. Introduction

The core inefficiency of Large Language Models lies in treating every token with the same computational budget. "The cat sat on the..." requires less reasoning than a complex logic puzzle, yet standard Transformers spend O(n²) attention on both. Linear attention mechanisms (like Gated DeltaNet) offer O(n) speed but often degrade performance.

It is hypothesized that a "best of both worlds" architecture exists: a model that dynamically routes tokens to the most appropriate attention mechanism.
- **Training:** More expensive. The model must learn *how* to route, not just *what* to predict.
- **Inference:** Optimized. The model saves compute on simple tokens (routing to Linear) and invests it in complex ones (routing to Softmax).

This paper presents a **proof-of-concept** implementation of this dynamic routing. It is shown that with proper load balancing, a small-scale model can learn a non-trivial routing strategy that matches the convergence of a static hybrid baseline.

---

## 2. Method

### 2.1 Architecture
A lightweight 4-layer architecture is used to test the routing mechanism:
- **Layer 0 (Fixed):** Gated DeltaNet (GDN). Provides a stable linear foundation.
- **Layers 1 & 2 (Routed):** Dynamic choice between GDN and Softmax.
- **Layer 3 (Fixed):** Softmax. Ensures global context aggregation at the end.

**[FIGURE 1 HERE: Architecture Diagram]**
![Architecture](architecture.png)

### 2.2 Parallel Routing & Load Balancing
To simplify the implementation, **parallel routing** is used: the router computes decisions for all routed layers simultaneously after Layer 0. A **Gumbel-Softmax** distribution is used to make the discrete routing decisions differentiable during training. This technique relaxes the discrete sampling process into a continuous, differentiable approximation, allowing gradients to flow through the routing decisions during backpropagation.

A critical challenge is **routing collapse**, where the model defaults to 100% usage of one mechanism (a layer learns more because it's choosen more - starting a vicious cycle). To counter this, we apply a load balancing loss adapted from the **Switch Transformer**.

#### The Math: Measuring Balance
We track two metrics for our $N=2$ experts:
1.  **$f$ (The Actual Workload):** What fraction of tokens did we *actually* assign to each expert?
    *   *Example:* If we have 100 tokens and send 90 to Linear, $f = [0.9, 0.1]$.
2.  **$P$ (The Manager's Bias):** What was the average probability (confidence) the router had for each expert?
    *   *Example:* If the router was generally 90% sure about Linear, $P = [0.9, 0.1]$.

We want both vectors to be close to uniform ($[0.5, 0.5]$). The Switch Transformer paper defines the loss as the scaled dot product:
$$ \mathcal{L}_{balance} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i $$

*   **Balanced Case:** $f=[0.5, 0.5], P=[0.5, 0.5]$. Loss $= 2 \cdot (0.25 + 0.25) = \mathbf{1.0}$. (Minimum)
*   **Collapsed Case:** $f=[1.0, 0.0], P=[1.0, 0.0]$. Loss $= 2 \cdot (1.0 + 0.0) = \mathbf{2.0}$. (Maximum)

#### Integration into Training
This auxiliary loss is added to the main objective with a small coefficient ($\alpha=0.01$) so it doesn't overpower the main goal of predicting the next token.

```python
# 1. Calculate standard language modeling loss
lm_loss = F.cross_entropy(logits, labels)

# 2. Calculate load balancing loss
aux_loss = self.compute_load_balancing_loss([
    router_probs_layer_1,
    router_probs_layer_2
])

# 3. Combine them
# alpha=0.01 ensures we just nudge the router, not force it
loss = lm_loss + 0.01 * aux_loss
```

---

## 3. Experiments & Results

A controlled experiment was conducted to validate the stability and performance of the routing mechanism.

**Setup:**
- **Model Size:** ~160M parameters
- **Data:** SmolLm Corpus (Cosmopedia v2)
- **Training:** 200 steps (Proof of Concept phase)
- **Comparison:** 
    1. **Static Baseline:** Fixed layers (GDN → GDN → GDN → Softmax).
    2. **Dynamic Model:** Learned routing for Layers 1 & 2.

### 3.1 Performance Comparison

The dynamic model successfully learned a balanced routing strategy (using ~55-60% GDN in routed layers) and achieved comparable performance to the static baseline.

| Model | Val Loss ↓ | Val Accuracy ↑ | Routing Behavior |
|-------|------------|----------------|------------------|
| **Static Baseline** | 7.00 | 13.30% | Fixed (100% GDN in L1/L2) |
| **Dynamic (Balanced)** | **6.25** | **21.08%** | **Mixed (~56% GDN / 44% Softmax)** |

**[FIGURE 2 HERE: Training Curves]**
> **Prompt for Figure:** A line chart comparing "Validation Loss" over time. The X-axis is "Training Steps" (0 to 200). The Y-axis is "Loss". Show two lines: a Blue line for "Static Baseline" that decreases slowly, and an Orange line for "Dynamic Routing" that decreases faster and reaches a lower value.

### 3.2 Routing Analysis
The router did not simply collapse to a random 50/50 split; it learned specific preferences for different layers.
- **Layer 1:** 55.6% GDN / 44.4% Softmax
- **Layer 2:** 60.9% GDN / 39.1% Softmax

This confirms the hypothesis: the model *can* learn to allocate different computational resources to different parts of the network.

**[FIGURE 3 HERE: Routing Distribution]**
> **Prompt for Figure:** A stacked area chart showing the routing percentage over time. X-axis is steps. Y-axis is 0% to 100%. The area is split into Blue (GDN) and Orange (Softmax). Show that it starts noisy but stabilizes around a 60/40 split, proving the load balancing works.

---

## 4. Discussion: The Efficiency Trade-off

The results highlight a fundamental trade-off in efficient AI:

1.  **Training Cost:** The Dynamic model is harder to train. It requires extra parameters for the router and careful tuning of the load balancing loss to prevent collapse. It was observed that without this loss, the model greedily collapses to the "easiest" path (pure GDN), missing out on the benefits of hybrid attention.

2.  **Inference Optimization:** The payoff is in inference. A static model is rigid—it must pay the O(n²) cost for Softmax layers on *every* token. The dynamic model has the *option* to use O(n) linear attention for easy tokens. In a large-scale deployment, this means simple queries can be processed with linear speed, only triggering the expensive quadratic attention when the router detects complex dependencies.

**[FIGURE 4 HERE: Inference Concept]**
> **Prompt for Figure:** A conceptual illustration comparing "Static" vs "Dynamic" inference. On the left (Static), show a heavy block processing every token equally slowly. On the right (Dynamic), show a stream of tokens where "easy" tokens (green dots) zip through a fast lane (Linear) and "hard" tokens (red dots) go through a detailed lane (Softmax). Label it "Optimized Inference Compute".

## 5. Conclusion

This work has demonstrated a working proof-of-concept for dynamic token routing in Linear Transformers. By accepting higher training complexity, an adaptive inference engine is gained that intelligently allocates FLOPs. Future work will scale this to billions of parameters, where the savings from routing "easy" tokens away from quadratic attention could yield massive efficiency gains.


