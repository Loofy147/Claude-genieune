# FINAL STEP: Populating the Kaggle Benchmark

Follow these manual steps to ensure your tasks appear in the Benchmark UI. The Kaggle CLI can push the code, but the final registration requires a UI action.

## 1. Register the Tasks on Kaggle
1. Go to [kaggle.com/benchmarks](https://www.kaggle.com/benchmarks).
2. Click **"Create task"** (Blue button, top right).
3. In the Notebook Editor:
   - **Paste the entire contents of** `genuineness_benchmark_production.py` into the first cell.
   - Click **"Save Version"** (top right) and wait for the run to complete.
   - **CRITICAL STEP**: After the version is saved, look at the top right of the editor or the Task Detail page. Click the **"Save Task"** button. This is what makes the tasks visible to your benchmark collection.

## 2. Populate the Benchmark Collection
1. Select **"Create benchmark"** on the [benchmarks home page](https://www.kaggle.com/benchmarks).
2. **Name:** `Genuineness Benchmark: Reasoning vs Pattern Completion`
3. **URL Slug:** `genuineness-benchmark`
4. **Description:** `A mechanistic suite identifying "genuine computation" heads in transformer architectures. It uses entropy variance and ablation to differentiate between dynamic reasoning and static retrieval.`
5. Click **"Add tasks"** and select the 5 tasks from your registered notebook:
   - `IOI Reasoning Accuracy`
   - `Genuine Head Density`
   - `Task Separation Score`
   - `Ablation Causal Impact`
   - `Output Genuineness Score`

## 3. Add Models to the Leaderboard
Click **"Add models"** and choose these recommended targets:
- `gpt2-xl`
- `meta-llama/Llama-3-8B`
- `mistralai/Mistral-7B-v0.1`
- `google/gemma-7b`

## Troubleshooting "Tasks Not Found"
- Ensure you clicked **"Save Task"** in the UI, not just "Save Version".
- Verify the notebook has **GPU** enabled in the right-hand settings panel before saving.
- Check that the notebook is **Public** or that you have permission to add its tasks to your benchmark.
