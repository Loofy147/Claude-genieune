import os, sys, subprocess, glob, json
from datetime import datetime

print("--- System Check: V3 Production ---")
# HUGGING_FACE_HUB_TOKEN is expected to be passed via the environment (e.g., Kaggle Secrets)

# Robust discovery for Kaggle datasets
engine_files = glob.glob("/kaggle/input/**/precision_targeting_engine.py", recursive=True)
if engine_files:
    for f in engine_files:
        path = os.path.dirname(f)
        if path not in sys.path:
            sys.path.append(path)
            print(f"Added engine path: {path}")

print("--- Installing dependencies ---")
# Fully relaxed dependency set to avoid resolution conflicts
subprocess.run(["pip", "install", "transformer-lens", "transformers", "torch", "einops", "--quiet"])

try:
    from precision_targeting_engine import run_full_pipeline
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
    print(f"Execution Error: {e}")
    import traceback
    traceback.print_exc()
