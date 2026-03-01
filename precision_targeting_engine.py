import torch
import numpy as np
import json
import re
import os
import subprocess
from datetime import datetime
from collections import defaultdict
from transformer_lens import HookedTransformer

# ══════════════════════════════════════════════════════════════════
# PRECISION TARGETING ENGINE (ACTUAL)
# ══════════════════════════════════════════════════════════════════

class PrecisionConstants:
    K_DEGRADE = 0.8129
    K_RECOVER = 1.2371
    GENUINE_DIFFUSE_VAR_H = 0.10
    GENUINE_DIFFUSE_COLLAPSES = 1
    IOI_ABLATION_THRESHOLD = 0.35

class RealTargetingEngine:
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        print(f"[ENGINE] Loading {model_name}...")
        self.model = HookedTransformer.from_pretrained(model_name)
        self.model.set_use_attn_result(True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def find_genuine_heads(self, prompts):
        head_stats = defaultdict(list)
        for prompt in prompts:
            tokens = self.model.to_tokens(prompt)
            _, cache = self.model.run_with_cache(tokens)
            for l in range(self.model.cfg.n_layers):
                pattern = cache[f"blocks.{l}.attn.hook_pattern"]
                for h in range(self.model.cfg.n_heads):
                    head_pattern = pattern[0, h]
                    entropy = -torch.sum(head_pattern * torch.log2(head_pattern + 1e-10), dim=-1)
                    max_h = np.log2(head_pattern.shape[-1])
                    norm_entropy = (entropy / max_h).cpu().numpy()
                    head_stats[(l, h)].append(norm_entropy)

        results = {}
        for (l, h), profiles in head_stats.items():
            all_entropies = np.concatenate(profiles)
            var_h = float(np.var(all_entropies))
            collapses = 0
            for profile in profiles:
                diffs = np.diff(profile)
                collapses += int(np.sum(diffs < -0.2))

            results[(l, h)] = {
                "var_h": var_h,
                "collapses": collapses,
                "is_genuine": var_h > PrecisionConstants.GENUINE_DIFFUSE_VAR_H and collapses >= PrecisionConstants.GENUINE_DIFFUSE_COLLAPSES
            }
        return [h for h, d in results.items() if d["is_genuine"]], results

class KaggleFullPipeline:
    def __init__(self, username="hichambedrani"):
        self.username = username
        self.project_name = "actual-precision-system"
        self.work_dir = "kaggle_deploy"
        os.makedirs(self.work_dir, exist_ok=True)

    def prepare_dataset(self):
        data_dir = os.path.join(self.work_dir, "dataset")
        os.makedirs(data_dir, exist_ok=True)
        meta = {"title": f"{self.project_name}-data", "id": f"{self.username}/{self.project_name}-data", "licenses": [{"name": "CC0-1.0"}]}
        with open(os.path.join(data_dir, "dataset-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        subprocess.run(["cp", "precision_targeting_engine.py", data_dir])
        return data_dir

    def prepare_notebook(self, use_gpu=True):
        kernel_dir = os.path.join(self.work_dir, "kernel")
        os.makedirs(kernel_dir, exist_ok=True)

        script_content = f"""
import os
import subprocess
print("Installing dependencies...")
subprocess.run(["pip", "install", "transformer-lens", "--quiet"])

import torch
from precision_targeting_engine import RealTargetingEngine

print(f"CUDA Available: {{torch.cuda.is_available()}}")
engine = RealTargetingEngine("gpt2")
prompts = ["John and Mary went to the store. John gave the apple to"]
genuine_heads, stats = engine.find_genuine_heads(prompts)

print(f"Found {{len(genuine_heads)}} genuine heads.")
for h in genuine_heads[:5]:
    print(f"Genuine Head: {{h}}")

# Save results
with open("benchmark_results.json", "w") as f:
    import json
    json.dump({{"genuine_heads": [str(h) for h in genuine_heads]}}, f)
"""
        with open(os.path.join(kernel_dir, "benchmark_run.py"), "w") as f:
            f.write(script_content)

        meta = {
            "id": f"{self.username}/{self.project_name}-notebook",
            "title": f"{self.project_name}-notebook",
            "code_file": "benchmark_run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if use_gpu else "false",
            "enable_internet": "true",
            "dataset_sources": [f"{self.username}/{self.project_name}-data"]
        }
        with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return kernel_dir

class KaggleHardwareInfo:
    @staticmethod
    def get_info():
        return {
            "GPU": "Tesla T4/P100 (16GB)",
            "TPU": "v3-8 (128GB)",
            "Models": "Llama-3, Gemma, Mistral, Qwen"
        }

if __name__ == "__main__":
    # Local check
    hw = KaggleHardwareInfo.get_info()
    print("Kaggle HW Info:", hw)

def test_run():
    engine = RealTargetingEngine("gpt2")
    prompts = ["The cat sat on the mat. The dog sat on the"]
    heads, stats = engine.find_genuine_heads(prompts)
    print(f"Heads found: {len(heads)}")

if __name__ == "__main__":
    test_run()
