# Experiment 11: Dynamic Per-Token Layer Routing

**Research Question**: Can a model dynamically choose between linear attention (GDN) and softmax attention per-token at each layer to achieve better performance than static layer assignments?

**Motivation**: Inference compute is the bottleneck for LLMs. If we can train models that dynamically use efficient linear attention when possible and expensive softmax attention only when needed, we can reduce inference costs significantly without sacrificing quality.

```python
cd experiments/exp11_dynamic_routing

# 1. Train baseline (static layers)
python run_experiment.py --config baseline

# 2. Train dynamic routing
python run_experiment.py --config dynamic

# 3. Compare results
python compare_experiments.py
```

---

## 🎯 Architecture

Both configurations use **4 layers total**:

### **Baseline (Static)**
```python
Layer 0: GDN (fixed)
Layer 1: GDN (fixed)
Layer 2: GDN (fixed)
Layer 3: Softmax (fixed)
```
- 75% linear attention, 25% softmax attention
- No routing overhead
- Baseline for comparison

### **Dynamic Routing**
```python
Layer 0: GDN (fixed) - stability
Layer 1: ROUTED per-token (GDN or Softmax)
Layer 2: ROUTED per-token (GDN or Softmax)
Layer 3: Softmax (fixed) - hypothesis from literature
```
- Layer 0 fixed to reduce variance
- Layer 3 fixed as softmax (based on research showing final layers benefit from full attention)
- Layers 1-2 can choose the best mechanism per-token

---

## 🔑 Key Features

### **Parallel Routing**
- All routing decisions made at layer 0 (after seeing input embeddings)
- Simpler to implement and reason about
- Faster than sequential routing

### **Load Balancing Loss**
Prevents routing collapse (model always choosing one mechanism):
```python
# Encourages ~50/50 split between GDN and Softmax for each layer
load_balance_loss = Σ (num_experts × fraction × probability)
total_loss = lm_loss + α × load_balance_loss  # α=0.01
```

### **Gumbel-Softmax for Differentiability**
- Allows backprop through discrete routing decisions
- Hard routing in forward pass (discrete)
- Soft routing in backward pass (differentiable)

---

## 📊 Expected Outcomes

**If Dynamic Routing Wins:**
- Validates adaptive attention hypothesis
- Shows different tokens need different mechanisms
- Path to more efficient inference

**If Baseline Wins:**
- Static layer assignment is sufficient
- Routing overhead not worth complexity
- Focus effort on better static hybrid ratios

---

## 🚀 Quick Start

### Train Baseline (Static)
```bash
cd experiments/exp11_dynamic_routing
python run_experiment.py --config baseline
```

### Train Dynamic Routing
```bash
python run_experiment.py --config dynamic
```

### Compare Results
```bash
python compare_experiments.py
```

---

## 📁 Configuration

**Model Size**: 
- Hidden size: 768
- Layers: 4
- Heads: 12
- Params: ~50M

**Training** (matches exp7 setup for fair comparison):
- Sequence length: 1024
- Batch size: 48
- Steps: 1000
- Learning rate: 1e-3 (baseline), 2e-3 (dynamic with attention)

---

## 📈 Metrics to Track

### Performance
- Validation loss (primary metric)
- Perplexity
- Tokens/second throughput

### Routing Statistics (Dynamic only)
- % tokens routed to GDN vs Softmax per layer
- Routing entropy (diversity of choices)
- Load balance loss value

---

## 🔬 Research Notes

**Why fix layer 0 as GDN?**
- Provides stable input processing
- Reduces routing variance
- Guaranteed linear attention efficiency

**Why fix layer 3 as Softmax?**
- Research suggests final layers benefit from global context
- Allows model to refine predictions with full attention before output

**Why route layers 1-2?**
- Middle layers are where adaptive computation may help most
- 2 routed layers = manageable complexity for first experiment
- Can scale to more layers if successful

---

## 📚 Related Work

- **Switch Transformer** (Google, 2021): Load balancing for MoE
- **Expert Choice Routing** (Google, 2022): Capacity-based routing
- **Mixture of Depths** (Google, 2024): Layer-wise adaptive computation
- **Exp7 (this repo)**: Static hybrid layer placement

---

## Files

- `config.py` - Both baseline and dynamic configurations
- `models.py` - Dynamic routing model implementation
- `run_experiment.py` - Training script for both configs
- `compare_experiments.py` - Compare baseline vs dynamic results
- `visualize_routing.py` - Analyze routing patterns
- `requirements.txt` - Dependencies
