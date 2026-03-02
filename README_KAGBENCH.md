# Genuineness Benchmark — Kaggle Setup Guide

## System Overview
This benchmark implements a mechanistic investigation into "Genuine Computation" across large language models. Unlike traditional benchmarks that measure output accuracy alone, this suite probes the underlying attention structure to differentiate between dynamic reasoning and static pattern retrieval.

### The Five Benchmark Tasks

| Task | Metric | Mechanism |
|------|--------|-----------|
| **IOI Reasoning Accuracy** | Accuracy | Measures how well the model identifies indirect objects in 25-35 token prompts. |
| **Genuine Head Density** | Fraction | Calculates the percentage of attention heads showing high entropy variance and verified collapse events. |
| **Reasoning vs Pattern Separation** | Delta Var | Measures the structural contrast in head activity between reasoning (IOI) and pattern (induction) tasks. |
| **Ablation Causal Impact** | Drop | Causal proof: measures the performance drop after mean-ablating the top identified "genuine" heads. |
| **Output Genuineness Score** | Score | A text-level classifier that scores model responses for commitment markers and lack of filler. |

## Kaggle Deployment Instructions

### 1. Create the Task Notebook
- Go to [Kaggle Benchmarks](https://www.kaggle.com/benchmarks) and select **"Create task"**.
- Upload or paste the contents of `genuineness_benchmark_tasks.py`.
- Ensure the environment settings have **GPU** enabled and **Internet** access.
- Save the task to register the functions with the `@kbench.task` decorator.

### 2. Configure the Benchmark
- Select **"Create benchmark"**.
- Title: `Genuineness Benchmark: Reasoning vs Pattern Completion`.
- Description: `A mechanistic suite to identify genuine computation heads in transformer architectures.`
- Add the five tasks created in the previous step.

### 3. Add Models for Evaluation
The benchmark supports models available in the Kaggle Community collection. Recommended targets:
- `gpt2-xl` (48 layers)
- `meta-llama/Llama-3-8B`
- `google/gemma-7b`
- `mistralai/Mistral-7B-v0.1`

### 4. Interpretation of Results
- **High Density + High Drop**: Strong evidence of a specialized reasoning circuit.
- **Low Density + Low Drop**: Evidence that the model is primarily relying on pattern matching / shallow retrieval.
- **High Separation**: Indicates that the model's architecture has physically distinct pathways for different types of computation.

## Integration with Repo
The tasks utilize the optimized `RealTargetingEngine` from `precision_targeting_engine.py`, which includes fixes for normalization bias and profile alignment.
