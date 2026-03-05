import os, sys, subprocess, glob, json
from datetime import datetime

print("--- System Check: V3 Production ---")
# HUGGING_FACE_HUB_TOKEN is expected to be passed via the environment (e.g., Kaggle Secrets)

engine_files = glob.glob("/kaggle/input/**/precision_targeting_engine.py", recursive=True)
if engine_files:
    sys.path.append(os.path.dirname(engine_files[0]))
    print(f"Engine path: {os.path.dirname(engine_files[0])}")

print("--- Installing dependencies ---")
# Fully relaxed dependency set to avoid resolution conflicts
subprocess.run(["pip", "install", "transformer-lens", "transformers", "torch", "einops", "--quiet"])

from precision_targeting_engine import run_full_pipeline

try:
    target = "gpt2-xl"
    print(f"--- Running Production Pipeline on {target} ---")
    results = run_full_pipeline(target)

    print(f"\nGenuine Heads: {results['genuine_heads_found']}")
    if results.get('ablation'):
        print(f"IOI Drop: {results['ablation']['drop']:.3f}")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("--- Execution Complete ---")
except Exception as e:
    import traceback
    traceback.print_exc()
