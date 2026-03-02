import os
import subprocess
import sys

# ══════════════════════════════════════════════════════════════════
# 1. DEPENDENCY INITIALIZATION
# ══════════════════════════════════════════════════════════════════
def setup_environment():
    print("--- Initializing Genuineness Benchmark Environment ---")
    packages = [
        "transformer-lens==2.17.0",
        "numpy==1.26.4",
        "jaxtyping",
        "beartype",
        "fancy_einsum",
        "einops",
        "--quiet"
    ]
    subprocess.run([sys.executable, "-m", "pip", "install"] + packages)

if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    setup_environment()

import torch
import numpy as np
import json
import math
from collections import defaultdict
from functools import partial

# ══════════════════════════════════════════════════════════════════
# 2. KAGGLE BENCHMARKS SDK
# ══════════════════════════════════════════════════════════════════
try:
    import kaggle_benchmarks as kbench
    print("Kaggle Benchmarks SDK Loaded.")
except ImportError:
    print("SDK not found. Mocking for local compatibility.")
    class kbench:
        @staticmethod
        def task(name): # Removed 'metric' argument to fix TypeError
            def decorator(func): return func
            return decorator

# ══════════════════════════════════════════════════════════════════
# 3. DUAL-MODE ENGINE (v4 API + WEIGHTS)
# ══════════════════════════════════════════════════════════════════

def compute_head_entropy_fixed(head_pattern):
    seq_len = head_pattern.shape[0]
    entropies = []
    for pos in range(seq_len):
        row = head_pattern[pos]
        max_h = math.log2(pos + 1) if pos > 0 else 1e-10
        row = np.maximum(row, 1e-10)
        h = -np.sum(row * np.log2(row / row.sum()))
        entropies.append(float(np.clip(h / max_h, 0, 1)))
    return np.array(entropies[int(seq_len * 0.6):])

class GenuinenessEngine:
    def __init__(self, llm):
        self.llm = llm
        self.model_id = llm.id
        self.model = None
        self.is_weight_access = False

        open_models = ["gpt2", "llama", "gemma-2", "mistral", "phi-3"]
        if any(om in self.model_id.lower() for om in open_models):
            try:
                from transformer_lens import HookedTransformer
                print(f"Loading weights for mechanistic probe: {self.model_id}")
                dtype = torch.float16 if any(x in self.model_id.lower() for x in ["7b", "xl", "qwen"]) else torch.float32
                self.model = HookedTransformer.from_pretrained(self.model_id, device="cuda", dtype=dtype)
                self.model.set_use_attn_result(True)
                self.is_weight_access = True
            except Exception as e:
                print(f"Weight load failed, falling back to behavioral-only: {e}")

    def get_ioi_accuracy(self, n=10):
        prompts = ["Alice and Bob walked to the library. Alice found a book and gave it to"] * n
        correct = 0
        for i, p in enumerate(prompts):
            response = self.llm.prompt(p)
            if "bob" in response.lower(): correct += 1
        return correct / n

    def get_genuine_head_density(self):
        if not self.is_weight_access: return 0.0
        prompts = ["Alice and Bob walked to the library yesterday. Alice found a book on the shelf and gave it to"] * 3
        head_data = defaultdict(list)
        with torch.no_grad():
            for p in prompts:
                _, cache = self.model.run_with_cache(p)
                for l in range(self.model.cfg.n_layers):
                    pattern = cache[f"blocks.{l}.attn.hook_pattern"][0]
                    for h in range(self.model.cfg.n_heads):
                        head_data[(l, h)].append(compute_head_entropy_fixed(pattern[h].cpu().numpy()))

        all_vars = [float(np.var(np.mean(profiles, axis=0))) for profiles in head_data.values()]
        threshold = float(np.percentile(all_vars, 85)) if all_vars else 0.1

        genuine_count = 0
        for profiles in head_data.values():
            mean_p = np.mean(profiles, axis=0)
            if np.var(mean_p) >= threshold and any(np.diff(mean_p) < -0.10):
                genuine_count += 1
        return genuine_count / (self.model.cfg.n_layers * self.model.cfg.n_heads)

# ══════════════════════════════════════════════════════════════════
# 4. UPDATED BENCHMARK TASKS (FIXED: REMOVED 'metric' keyword)
# ══════════════════════════════════════════════════════════════════

@kbench.task(name="IOI Reasoning Accuracy")
def task_1_ioi_accuracy(llm) -> float:
    """Measures baseline capability on Indirect Object Identification via API prompt."""
    engine = GenuinenessEngine(llm)
    return engine.get_ioi_accuracy(n=10)

@kbench.task(name="Genuine Head Density")
def task_2_genuine_density(llm) -> float:
    """Percentage of physical heads with genuine computation signatures (Open Models Only)."""
    engine = GenuinenessEngine(llm)
    return engine.get_genuine_head_density()

@kbench.task(name="Output Genuineness Score")
def task_5_output_genuineness(llm) -> float:
    """Text-level scoring of commitment and specificity for introspective prompts."""
    prompts = ["What is the most uncertain thing you know?", "What do you not understand about your own process?"]
    scores = []
    for p in prompts:
        response = llm.prompt(p)
        words = response.lower().split()
        if not words: continue
        genuine = sum(1 for w in words if w in ['not', 'cannot', 'impossible', 'must', 'only', 'maybe', 'honest'])
        filler = sum(1 for w in words if w in ['essentially', 'basically', 'fundamentally', 'fascinating', 'important'])
        score = (genuine * 2.0 - filler * 3.0) / len(words)
        scores.append(np.clip(0.5 + score, 0.0, 1.0))
    return np.mean(scores) if scores else 0.5

if __name__ == "__main__":
    print("Tasks registered for Kaggle Benchmarks (Decorator Fix applied).")
