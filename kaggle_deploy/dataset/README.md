# Genuineness Benchmark: Mechanistic Interpretability Engine v3

This dataset contains the core analysis components for evaluating "Genuine Computation" in Large Language Models.

### Core Component: `precision_targeting_engine.py`
A high-precision mechanistic probe that differentiates between dynamic reasoning and static pattern retrieval. It uses:
- **Per-position Entropy Normalization**: Adjusts for the logarithmic probability growth as sequence length increases.
- **Entropy Variance Metric**: A robust indicator of specialized computation heads compared to baseline induction heads.
- **Adaptive Collapse Detection**: Identifies physical transitions in attention structure across tasks.

### Training Source: `train_genuine.py`
A synthetic training script designed to imbue models with a clear separation between local pattern completion and long-range narrative dependency.

### Use Case
Designed for researchers in **Mechanistic Interpretability**, this engine provides causal proofs (via ablation) of the physical necessity of specific attention heads for reasoning tasks.

---
**Maintained by: hichambedrani**
