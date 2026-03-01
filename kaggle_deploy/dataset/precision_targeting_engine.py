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
# PRECISION TARGETING ENGINE
# ══════════════════════════════════════════════════════════════════

class PrecisionConstants:
    K_DEGRADE = 0.8129
    K_RECOVER = 1.2371
    GENUINE_DIFFUSE_VAR_H = 0.10
    GENUINE_DIFFUSE_COLLAPSES = 1
    IOI_ABLATION_THRESHOLD = 0.35

class PromptGenerator:
    """Generates diverse reasoning prompts for benchmarking."""

    @staticmethod
    def generate_ioi(names=("Alice", "Bob", "Charlie"), object="book"):
        n1, n2, n3 = names
        return f"{n1} and {n2} went to the library. {n1} gave the {object} to"

    @staticmethod
    def generate_syllogism():
        templates = [
            ("All A are B. All B are C. Therefore, all A are", "C"),
            ("Some A are B. All B are C. Therefore, some A are", "C"),
            ("No A are B. All C are A. Therefore, no C are", "B")
        ]
        return templates[np.random.randint(len(templates))]

    @staticmethod
    def generate_counterfactual_ioi():
        # Test if the model actually reasons or just completes the pattern
        return "In a world where gravity pushes up, Alice drops a ball. The ball will"

    @staticmethod
    def get_benchmark_library():
        return {
            "ioi": [PromptGenerator.generate_ioi(names=("John", "Mary", "John"), object="apple"),
                    PromptGenerator.generate_ioi(names=("Alice", "Bob", "Alice"), object="key")],
            "syllogism": [PromptGenerator.generate_syllogism()[0]],
            "counterfactual": [PromptGenerator.generate_counterfactual_ioi()]
        }

class DatasetManager:
    """Manages integration with external Kaggle datasets."""
    def __init__(self):
        self.datasets = {
            "gsm8k": "thedevastator/grade-school-math-8k-q-a",
            "cot": "konradb/chain-of-thought-collection",
            "aime": "dolbokostya/math-problems-with-answers-aime-imo"
        }

    def get_source_list(self):
        return list(self.datasets.values())

    def download_sample(self, dataset_key):
        ref = self.datasets.get(dataset_key)
        if ref:
            print(f"[DATASET] Simulating download of {ref}...")
            return f"Sample data from {ref}"
        return None

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
        self.project_name = "enhanced-precision-system"
        self.work_dir = "kaggle_deploy"
        self.dataset_manager = DatasetManager()
        os.makedirs(self.work_dir, exist_ok=True)

    def prepare_dataset(self):
        data_dir = os.path.join(self.work_dir, "dataset")
        os.makedirs(data_dir, exist_ok=True)

        # Save prompt library
        prompts = PromptGenerator.get_benchmark_library()
        with open(os.path.join(data_dir, "prompt_library.json"), "w") as f:
            json.dump(prompts, f, indent=2)

        meta = {"title": f"{self.project_name}-data", "id": f"{self.username}/{self.project_name}-data", "licenses": [{"name": "CC0-1.0"}]}
        with open(os.path.join(data_dir, "dataset-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        subprocess.run(["cp", "precision_targeting_engine.py", data_dir])
        return data_dir

    def prepare_notebook(self, use_gpu=True):
        kernel_dir = os.path.join(self.work_dir, "kernel")
        os.makedirs(kernel_dir, exist_ok=True)

        sources = [f"{self.username}/{self.project_name}-data"] + self.dataset_manager.get_source_list()

        meta = {
            "id": f"{self.username}/{self.project_name}-notebook",
            "title": f"{self.project_name}-notebook",
            "code_file": "benchmark_run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if use_gpu else "false",
            "enable_internet": "true",
            "dataset_sources": sources
        }
        with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return kernel_dir

if __name__ == "__main__":
    # Local check of prompt generation
    print("Generated IOI:", PromptGenerator.generate_ioi())
    print("Prompt Library Keys:", list(PromptGenerator.get_benchmark_library().keys()))
    dm = DatasetManager()
    print("Dataset Sources:", dm.get_source_list())
