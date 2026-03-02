# Kaggle Benchmark Population Guide

Follow these steps to create and populate the benchmark on the Kaggle UI.

## 1. Create the Benchmark
Go to [kaggle.com/benchmarks](https://www.kaggle.com/benchmarks) and click **"Create benchmark"**.

**Name:**
`Genuineness Benchmark: Reasoning vs Pattern Completion`

**Description:**
`A mechanistic suite identifying "genuine computation" heads in transformer architectures. Unlike accuracy-only benchmarks, this probes the underlying attention structure to differentiate between dynamic reasoning and static pattern retrieval.`

---

## 2. Add Tasks
Once the benchmark is created, add the tasks from your notebook `hichambedrani/genuineness-benchmark-v3-tasks`.

| Task Name | Description | Metric |
|-----------|-------------|--------|
| **IOI Reasoning Accuracy** | Baseline accuracy on Indirect Object Identification prompts. | `accuracy` |
| **Genuine Head Density** | Percentage of attention heads showing dynamic entropy variance + collapse. | `fraction` |
| **Reasoning vs Pattern Separation** | Structural contrast in head dynamics between reasoning and induction tasks. | `delta_var` |
| **Ablation Causal Impact** | Performance drop after mean-ablating top identified genuine heads. | `drop` |
| **Output Genuineness Score** | Text-level classification of introspective model outputs. | `score` |

---

## 3. Add Models
Add the following models from the Community list to build the leaderboard:
- `gpt2-xl`
- `meta-llama/Llama-3-8B`
- `mistralai/Mistral-7B-v0.1`
- `google/gemma-7b`

---

## 4. Why This Benchmark?
This benchmark implements an **Inverted Asymmetry** framework where we verify that recovery ($k_{recover}$) is faster than degradation ($k_{degrade}$), and reasoning circuits exist in a specific "GENUINE_DIFFUSE" phase space. It provides architecture-level causal proof of reasoning specialized circuits.
