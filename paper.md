Auto-selecting linear vs full (softmax) attention layers per token

Issue: model colapse

inference takes a lot more compute for the LLM then the training, if we can create models that inference more efficiently, the increase in training time doesn't matter.


# 4 layers total:
Layer 0: FIXED as GDN (stability in early layers)
Layer 1: ROUTED (GDN or Softmax)
Layer 2: ROUTED (GDN or Softmax)  
Layer 3: FIXED as Softmax (you said this)

I want to keep first layer fixed to reduce variation and make it easier to experiment and learn. Last layer is softmax due to a paper.


---

# Decide ALL routes at the beginning based on input embeddings
router_logits = router_network(input_embeddings)  # [batch, seq, num_routed_layers, 2]

# Shape: [batch, seq, 2, 2] for layers 1 and 2
# routes[..., 0, :] = routing for layer 1
# routes[..., 1, :] = routing for layer 2

✅ Simpler to implement
✅ Can be computed once at the start
✅ Easier to reason about load balancing
✅ Faster (parallel routing decision)