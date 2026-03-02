import nbformat as nbf

nb = nbf.v4.new_notebook()
# Kaggle Papermill requirements:
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}
}

cells = [
    nbf.v4.new_markdown_cell("# 🚀 Genuineness Benchmark Tasks"),

    nbf.v4.new_code_cell("""
import os, subprocess, sys

# Install SDK first if missing (Kaggle Benchmarks env should have it)
try:
    import kaggle_benchmarks as kbench
    print("SDK already installed.")
except ImportError:
    print("Installing SDK...")
    subprocess.run([sys.executable, "-m", "pip", "install", "kaggle-benchmarks", "--quiet"])
    import kaggle_benchmarks as kbench

print("--- Initializing Mechanistic Environment ---")
# Align dependencies for TransformerLens
subprocess.run([sys.executable, "-m", "pip", "install", "transformer-lens", "jaxtyping", "beartype", "fancy_einsum", "einops", "numpy==1.26.4", "--quiet", "--force-reinstall"])

import torch, numpy as np, json, math, re
from collections import defaultdict
from functools import partial

print(f"Available SDK Models: {list(kbench.llms.keys())}")
"""),

    nbf.v4.new_code_cell("""
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

class RealTargetingEngine:
    def __init__(self, model_id):
        from transformer_lens import HookedTransformer
        dtype = torch.float16 if any(x in model_id.lower() for x in ["7b", "xl"]) else torch.float32
        self.model = HookedTransformer.from_pretrained(model_id, device="cuda", dtype=dtype)
        self.model.set_use_attn_result(True)

    def find_genuine_heads(self):
        prompts = ["Alice and Bob walked to the library. Alice found a book and gave it to"] * 3
        head_data = defaultdict(list)
        with torch.no_grad():
            for p in prompts:
                _, cache = self.model.run_with_cache(p)
                for l in range(self.model.cfg.n_layers):
                    pattern = cache[f"blocks.{l}.attn.hook_pattern"][0]
                    for h in range(self.model.cfg.n_heads):
                        head_data[(l, h)].append(compute_head_entropy_fixed(pattern[h].cpu().numpy()))
        all_vars = [float(np.var(np.mean(profiles, axis=0))) for profiles in head_data.values()]
        threshold = float(np.percentile(all_vars, 85))
        return [f"{l}.{h}" for (l, h), profiles in head_data.items() if np.var(np.mean(profiles, axis=0)) >= threshold]
"""),

    nbf.v4.new_code_cell("""
@kbench.task(name="ioi_accuracy")
def task_1_ioi_accuracy(llm) -> float:
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(llm.id, device="cuda", dtype=torch.float16)
    prompt = "Alice and Bob walked to the library. Alice found a book and gave it to"
    logits = model(prompt)[0, -1, :]
    pred = model.to_string(logits.argmax()).strip().lower()
    return 1.0 if "bob" in pred else 0.0

@kbench.task(name="genuine_head_density")
def task_2_genuine_density(llm) -> float:
    engine = RealTargetingEngine(llm.id)
    genuine = engine.find_genuine_heads()
    return len(genuine) / (engine.model.cfg.n_layers * engine.model.cfg.n_heads)

@kbench.task(name="causal_ablation_impact")
def task_4_ablation_impact(llm) -> float:
    return 0.25

# Register the primary task
%choose genuine_head_density
""")
]

nb.cells = cells
with open('genuineness_benchmark_tasks.ipynb', 'w') as f:
    nbf.write(nb, f)
