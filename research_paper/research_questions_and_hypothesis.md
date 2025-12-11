# Research Questions and Hypotheses

## Core Goal
To understand the theoretical and practical underpinnings of the **Muon optimizer**, specifically why its spectral normalization of gradients leads to superior convergence properties compared to traditional adaptive methods like AdamW.

---

## 1. The Role of Singular Value Spectra
**Question**: Muon produces significantly flatter singular value spectra in weight matrices compared to AdamW (which shows steep decay). Is this "feature equality" essential for its fast convergence?
**Hypothesis**: By maintaining a flatter spectrum, Muon prevents the "rank collapse" seen in AdamW early in training. This ensures that gradients continue to flow through more dimensions of the latent space, accelerating learning in the initial phases.

## 2. Gradient vs. Weight Orthogonalization
**Question**: Muon orthogonalizes *updates* (gradients) via Newton-Schulz, whereas manifold methods (like L1-Stiefel) enforce orthogonality on the *weights*. Which approach is more effective for deep learning?
**Hypothesis**: Orthogonalizing updates (Muon) is superior because it conditions the optimization trajectory (preconditioning) without constraining the final solution space. Strict weight orthogonality (Stiefel) may be too restrictive, hindering the model's ability to reach optimal solutions that require non-orthogonal weight structures.

## 3. Scaling to Large Models (LLMs)
**Question**: Do the matrix properties observed in MNIST (flat spectra, fast convergence) hold for high-dimensional Transformers?
**Hypothesis**: The benefits of Muon will amplify at scale. In Deep Transformers, the "vanishing signal" problem is often linked to collapsing singular values. Muon's ability to maintain rank/flat spectra will lead to even greater relative speedups in LLM pre-training compared to small-scale vision tasks.