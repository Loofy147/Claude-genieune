
import os
import sys
import subprocess
import glob

print("--- System Check ---")
print(f"Current Directory: {os.getcwd()}")

# Search for the engine file
engine_files = glob.glob("/kaggle/input/**/precision_targeting_engine.py", recursive=True)
if engine_files:
    engine_path = os.path.dirname(engine_files[0])
    print(f"Adding engine path to sys.path: {engine_path}")
    sys.path.append(engine_path)
else:
    print("WARNING: precision_targeting_engine.py not found yet. Listing /kaggle/input:")
    for root, dirs, files in os.walk("/kaggle/input"):
        for file in files:
            if file == "precision_targeting_engine.py":
                print(f"Found it at: {root}")
                sys.path.append(root)

print("--- Installing dependencies ---")
# Try installing transformer-lens without strict versioning
subprocess.run(["pip", "install", "transformer-lens", "jaxtyping", "beartype", "fancy_einsum", "einops", "--quiet"])

import torch
import numpy
print(f"Numpy Version: {numpy.__version__}")

try:
    from precision_targeting_engine import RealTargetingEngine
    print("--- Engine Imported Successfully ---")

    print(f"CUDA Available: {torch.cuda.is_available()}")
    engine = RealTargetingEngine("gpt2")
    prompts = ["John and Mary went to the store. John gave the apple to"]
    genuine_heads, stats = engine.find_genuine_heads(prompts)

    print(f"Found {len(genuine_heads)} genuine heads.")
    with open("benchmark_results.json", "w") as f:
        import json
        json.dump({"genuine_heads": [str(h) for h in genuine_heads]}, f)

except Exception as e:
    print(f"--- Error during execution: {e} ---")
    import traceback
    traceback.print_exc()
