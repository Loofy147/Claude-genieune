# FINAL STEP: Populating the Genuineness Benchmark

This guide includes the latest SOTA models available via the Kaggle Benchmarks SDK (Claude 4.6, Gemini 3, Gemma 3, DeepSeek v3.2).

## 1. Register the Tasks on Kaggle
1. Go to [kaggle.com/benchmarks](https://www.kaggle.com/benchmarks).
2. Click **"Create task"**.
3. Paste the entire contents of `genuineness_benchmark_production.py`.
4. Click **"Save Version"**.
5. After saving, click the **"Save Task"** button on the Task Detail page.

## 2. Assemble the Benchmark
1. Create a new benchmark named `Genuineness Benchmark: Global Reasoning Leaderboard`.
2. Add these tasks from your notebook:
   - `IOI Reasoning Accuracy` (Behavioral)
   - `Genuine Head Density` (Mechanistic - Open Weights only)
   - `Output Genuineness Score` (Behavioral)

## 3. Recommended Models to Compare
Add these to the leaderboard to compare API-only vs Open-weights performance:

### API-Only (Behavioral Tasks)
- `anthropic/claude-opus-4-6@default`
- `anthropic/claude-sonnet-4-6@default`
- `google/gemini-3.1-pro-preview`
- `deepseek-ai/deepseek-v3.2`

### Open Weights (Mechanistic + Behavioral)
- `google/gemma-3-27b`
- `qwen/qwen3-next-80b-a3b-thinking`
- `meta-llama/Llama-3-8B`

## 4. Understanding the 'Dual-Mode' Results
- **API Models**: Will score 0.0 or N/A on "Genuine Head Density" because their weights are not accessible for TransformerLens. Their "genuineness" is measured purely via Task 1 and 5.
- **Open Models**: Provide the full picture, showing if high accuracy (Task 1) correlates with actual physical reasoning circuits (Task 2).
