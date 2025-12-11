# Experiment Plan: Oblique vs L1-Stiefel Matrix Dynamics

## 1. Objective
To understand the "geometric inside" of the weight matrices learned by **Oblique** (unit-norm columns) versus **L1-Stiefel** (orthonormal columns + sparsity).

**Core Question:** Why does Oblique achieve lower loss (0.027) compared to L1-Stiefel (0.115) despite Oblique suffering from massive rank collapse?

## 2. Hypothesis
**"The Feature Redundancy Hypothesis"**: we suspect that for tasks like MNIST, strict orthogonality (Stiefel) is too restrictive. It forces every neuron to learn a completely unique feature. In contrast, Oblique allows neurons to be **correlated**, enabling the network to "gang up" on difficult features or create robust ensembles for noisy signals.

## 3. Methodology: The Gram Matrix Scan
We will analyze the Gram Matrix $G = W^T W$ during training. This matrix contains the dot products of every column pair.

### A. Metrics to Track
We will instrument the training loop to capture these metrics every 50 steps:

1.  **Off-Diagonal Mass (Feature Correlation)**:
    *   Formula: $\frac{1}{N(N-1)} \sum_{i \neq j} |<w_i, w_j>|$
    *   *What it tells us:* How "redundant" are the learned features?
    *   *Prediction:* High for Oblique, Low/Zero for Stiefel.

2.  **Orthogonality Error**:
    *   Formula: $||W^T W - I||_F$
    *   *What it tells us:* How far the matrix is from being a rigid rotation.
    *   *Prediction:* High for Oblique, Low for Stiefel.

3.  **Sparsity**:
    *   Formula: Fraction of weights with $|w_{ij}| < 0.01$.
    *   *Note:* L1-Stiefel explicitly optimizes for this; Oblique does not.

### B. Visualization: The "Fingerprint" Histogram
At the end of training, we will compute the pairwise cosine similarity of all column pairs and plot the distribution.

*   **L1-Stiefel Fingerprint**: Should be a Dirac delta function at x=0 (all features orthogonal).
*   **Oblique Fingerprint**: Should be a broad distribution (some orthogonal, some highly correlated).

## 4. Implementation Details
*   **Network**: Standard MLP (784 -> 256 -> 128 -> 10)
*   **Target Layer**: We will focus analysis on `fc1` (the first hidden layer, 784x256), as this learns the primary features from pixels.
*   **Code Location**: `mnist/experiments/oblique_vs_stiefel.py`

## 5. Success Criteria
The experiment is successful if we can visualize a clear trade-off: **Stiefel buys stability and orthogonality at the price of feature redundancy, whereas Oblique buys lower loss at the price of rank collapse.**
