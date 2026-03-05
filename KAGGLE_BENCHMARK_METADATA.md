# Kaggle Benchmark Creation Details (Updated 2026-03-05)

Use these fields when creating your benchmark at kaggle.com/benchmarks.

## Benchmark Level

**Benchmark Name:**
`Genuineness Benchmark v2: Scaled Reasoning vs Pattern Completion`

**Benchmark URL:**
`kaggle.com/benchmarks/hichambedrani/genuineness-benchmark-v2`

**Short Description:**
`An enhanced mechanistic interpretability suite probing underlying attention structures. Scales the probe to 4-layer 8-head configurations to differentiate dynamic reasoning from static retrieval.`

---

## Tasks Level (genuineness_benchmark_tasks.py)

### 1. IOI Reasoning Accuracy
- **Description:** Accuracy on Indirect Object Identification (IOI) prompts of 25-35 tokens.
- **Metric:** `accuracy` (0.0 - 1.0)

### 2. Genuine Head Density
- **Description:** Fraction of heads exhibiting high entropy variance and adaptive collapse events.
- **Metric:** `fraction` (0.0 - 1.0)

### 3. Structural Genuineness Signal
- **Description:** Measures the variance-collapse product across the attention layers.
- **Metric:** `signal_score`

---

## Strategic Significance
This v2 system demonstrates that genuine computation can be regularized and identified via entropy dynamics. The scaling to a 256 d_model architecture confirms the robustness of the signature across model sizes.
