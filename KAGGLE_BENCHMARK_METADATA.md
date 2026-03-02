# Kaggle Benchmark Creation Details

Use these fields when creating your benchmark at kaggle.com/benchmarks.

## Benchmark Level

**Benchmark Name:**
`Genuineness Benchmark: Reasoning vs Pattern Completion`

**Benchmark URL:**
`kaggle.com/benchmarks/hichambedrani/genuineness-benchmark`

**Short Description:**
`A mechanistic interpretability suite that probes the underlying attention structure of transformers to differentiate between dynamic reasoning (genuine computation) and static retrieval (pattern completion).`

---

## Tasks Level (genuineness_benchmark_tasks.py)

### 1. IOI Reasoning Accuracy
- **Description:** Measures the model's accuracy on Indirect Object Identification (IOI) prompts of 25-35 tokens. It verifies if the model can correctly identify the recipient of an object in a narrative context.
- **Metric:** `accuracy` (0.0 - 1.0)

### 2. Genuine Head Density
- **Description:** Calculates the percentage of attention heads that exhibit the 'genuine' signature: high entropy variance (p85 threshold) and at least one verified collapse event during reasoning.
- **Metric:** `fraction` (0.0 - 1.0)

### 3. Reasoning vs Pattern Separation
- **Description:** Measures the structural contrast by calculating the difference in entropy variance between reasoning tasks (IOI) and simple induction tasks (pattern repetition).
- **Metric:** `delta_var`

### 4. Ablation Causal Impact
- **Description:** Provides causal proof by measuring the drop in IOI performance after mean-ablating the top-5 identified "genuine" heads. Selective drop confirms these heads are necessary for reasoning.
- **Metric:** `drop`

### 5. Output Genuineness Score
- **Description:** A text-level classifier that scores the model's self-generated responses to introspective prompts for commitment markers, specificity, and lack of rote filler text.
- **Metric:** `score` (0.0 - 1.0)

---

## Strategic Significance
This benchmark builds what the data implied: an inverted asymmetry where recovery is faster than degradation, and genuine computation exists in a distinct "GENUINE_DIFFUSE" phase space. By running this across all Kaggle models, we create a definitive leaderboard of architecture-level reasoning genuineness.
