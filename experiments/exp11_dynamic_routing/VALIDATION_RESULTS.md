# Experiment 11: Dynamic Routing - Quick Validation Results (29 steps)

## Summary

Successfully validated both baseline and dynamic routing configurations with 29 training steps each.

## Results

### Baseline (Static Layers: GDN 0-2, Softmax 3)
- **Parameters**: 141.7M
- **Best Val Loss**: 7.4473
- **Val Accuracy**: 10.40%
- **Val Perplexity**: 1,715.31
- **Training Time**: 8.2s
- **Architecture**: Fixed [GDN, GDN, GDN, Softmax]

### Dynamic Routing (Layer 0: GDN, Layers 1-2: Routed, Layer 3: Softmax)
- **Parameters**: 160.6M (+18.9M for routing)
- **Best Val Loss**: 7.6007 (slightly worse)
- **Val Accuracy**: 8.87%
- **Val Perplexity**: 1,999.52
- **Training Time**: 9.3s
- **Routing Behavior** (at step 20):
  - Layer 1: GDN=51.2%, Softmax=48.8% (nearly balanced)
  - Layer 2: GDN=64.3%, Softmax=35.7% (prefers GDN)

## Key Observations

1. ✅ **Both configs work end-to-end** - No crashes, proper training/eval
2. 🔄 **Routing is active** - Dynamic model learns to route differently per layer
3. 📊 **Baseline slightly better** - But this is only 29 steps, not enough to conclude
4. 🎯 **Router learns patterns** - Layer 2 prefers GDN (64%), Layer 1 balanced
5. ⚡ **Fast iteration** - ~9s per config with caching enabled

## Technical Fixes Applied

1. Fixed broken import statement
2. Installed missing dependencies (einops, flash-attn)
3. Implemented data caching (10M tokens → instant reload)
4. Fixed dtype mismatches in routing
5. Corrected FLA attribute names (embeddings vs embed_tokens)
6. Reduced batch size to avoid OOM (48→16)

## Next Steps

Ready to run full experiments with more steps. The 29-step validation confirms:
- Data pipeline works
- Both models train correctly
- Routing mechanism functions
- Evaluation and checkpointing operational

## Recommendations for Full Run

1. Increase steps to 1000-5000 for meaningful comparison
2. Monitor routing statistics throughout training
3. Compare training curves baseline vs dynamic
4. Analyze which layers learn to prefer which attention type
