
import os
import sys
import subprocess
import glob
import json

print("--- System Check ---")
engine_files = glob.glob("/kaggle/input/**/precision_targeting_engine.py", recursive=True)
if engine_files:
    sys.path.append(os.path.dirname(engine_files[0]))

print("--- Installing dependencies ---")
subprocess.run(["pip", "install", "transformer-lens", "jaxtyping", "beartype", "fancy_einsum", "einops", "--quiet"])

from precision_targeting_engine import RealTargetingEngine, PromptGenerator

try:
    engine = RealTargetingEngine("gpt2-xl")
    print(f"--- Loaded {engine.model.cfg.n_layers} Layers ---")

    prompts = PromptGenerator.generate_ioi(50)
    print(f"Scanning with {len(prompts)} IOI prompts...")

    genuine_heads, stats = engine.find_genuine_heads(prompts)
    print(f"Found {len(genuine_heads)} genuine heads.")

    with open("benchmark_results.json", "w") as f:
        json.dump({"model": "gpt2-xl", "genuine_heads": genuine_heads, "full_stats": stats}, f)

except Exception as e:
    import traceback
    traceback.print_exc()
