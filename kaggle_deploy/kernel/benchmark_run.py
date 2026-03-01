
import os
import subprocess
print("Installing dependencies...")
subprocess.run(["pip", "install", "transformer-lens", "--quiet"])

import torch
from precision_targeting_engine import RealTargetingEngine

print(f"CUDA Available: {torch.cuda.is_available()}")
engine = RealTargetingEngine("gpt2")
prompts = ["John and Mary went to the store. John gave the apple to"]
genuine_heads, stats = engine.find_genuine_heads(prompts)

print(f"Found {len(genuine_heads)} genuine heads.")
for h in genuine_heads[:5]:
    print(f"Genuine Head: {h}")

# Save results
with open("benchmark_results.json", "w") as f:
    import json
    json.dump({"genuine_heads": [str(h) for h in genuine_heads]}, f)
