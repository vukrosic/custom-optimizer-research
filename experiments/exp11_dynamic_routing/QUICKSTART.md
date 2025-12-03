# Experiment 11: Dynamic Routing - Quick Start

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd experiments/exp11_dynamic_routing
pip install -r requirements.txt
```

Make sure `flash-linear-attention` submodule is cloned:
```bash
cd ../../
git submodule update --init --recursive
```

### 2. Train Baseline (Static)
```bash
python run_experiment.py --config baseline
```

This trains a 4-layer model: [GDN, GDN, GDN, Softmax]

### 3. Train Dynamic Routing
```bash
python run_experiment.py --config dynamic
```

This trains a 4-layer model: [GDN, ROUTED, ROUTED, Softmax]

###4. Compare Results
```bash
python compare_experiments.py
```

---

## 📊 Expected Runtime

- **H100**: ~15-20 minutes per config (1000 steps)
- **A100**: ~25-30 minutes per config
- **RTX 4090**: ~40-50 minutes per config

---

## 🔧 Troubleshooting

### Routing Collapse
If dynamic routing always chooses one mechanism (>90%), try:
```bash
python run_experiment.py --config dynamic_aggressive
```

This uses stronger load balancing (alpha=0.05 instead of 0.01).

### Out of Memory
Reduce batch size in `config.py`:
```python
batch_size=24  # Instead of 48
```

---

## 📈 What to Look For

**Baseline should be stable** - it's just a standard model.

**Dynamic routing success indicators:**
- Validation loss < Baseline loss (even by 1-2%)
- Routing is balanced (~40-60% split per layer)
- Load balance loss decreases over training

**Routing collapse indicators:**
- One mechanism used >90% of the time
- Load balance loss stays high
- Dynamic worse than baseline

---

## 🎯 Next Steps

If dynamic routing works:
1. Try routing more layers
2. Test on longer contexts
3. Analyze which tokens prefer which mechanism
4. Try per-head routing instead of per-layer

If baseline wins:
1. Static layer assignment is sufficient
2. Focus on Exp7 (which static ratio is best)
3. Routing overhead not worth complexity
