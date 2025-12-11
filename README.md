# Motivation

Making any progress in neural network optimizers will have massive ripple effects on every AI / neural network in the world.

This research aims to do exactly that.

To come up with new optimizers, we shoudl focus on understanding why current ones work, as opposed to coming up with new ideas. Ideas will be a lot better once we understand why current ones work.

In this work we will do extensive experiments to gain practical and theoretical understanding of optimizers, especially why the new best Muon optimizer outperforms other optimizers.

We will use 2 experiments: MNIST and LLM. Each will have different optimizers or optimizer components applied and tested.

1. AdamW
2. Muon
3. Manifold Muon
4. Manifold Muon + Stiefel


python -m mnist.experiments.optimizer_comparison --epochs 5
 - Compares all optimizers on MNIST