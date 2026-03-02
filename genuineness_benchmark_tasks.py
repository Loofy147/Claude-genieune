import torch
import numpy as np
import json
import os
import re
from collections import defaultdict
from functools import partial

try:
    import kaggle_benchmarks as kbench
except ImportError:
    class kbench:
        @staticmethod
        def task(name, metric):
            def decorator(func):
                func.task_name = name
                func.metric = metric
                return func
            return decorator

from precision_targeting_engine import RealTargetingEngine, PromptGenerator, compute_head_entropy_fixed, detect_collapses

# ══════════════════════════════════════════════════════════════════
# KAGGLE BENCHMARK TASKS
# ══════════════════════════════════════════════════════════════════

@kbench.task(name="IOI Reasoning Accuracy", metric="accuracy")
def task_1_ioi_accuracy(model_id):
    """Task 1: Baseline reasoning capability."""
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(model_id, device="cuda", dtype=torch.float16)
    prompts = PromptGenerator.generate_ioi(20)
    correct = 0
    for p in prompts:
        tokens = model.to_tokens(p)
        with torch.no_grad(): logits = model(tokens)[0, -1, :]
        p2 = p.split()[2]
        pred = model.to_string(logits.argmax())
        if p2.strip().lower() in pred.strip().lower():
            correct += 1
    return correct / len(prompts)

@kbench.task(name="Genuine Head Density", metric="fraction")
def task_2_genuine_density(model_id):
    """Task 2: Fraction of heads with genuine computation signature."""
    engine = RealTargetingEngine(model_id)
    prompts = PromptGenerator.generate_ioi(10)
    genuine_heads, stats = engine.find_genuine_heads(prompts)
    total = engine.model.cfg.n_layers * engine.model.cfg.n_heads
    return len(genuine_heads) / total

@kbench.task(name="Reasoning vs Pattern Separation", metric="delta_var")
def task_3_separation(model_id):
    """Task 3: Difference in entropy variance between reasoning and induction."""
    engine = RealTargetingEngine(model_id)
    r_prompts = PromptGenerator.generate_ioi(5)
    p_prompts = PromptGenerator.generate_induction(5)

    def get_avg_var(prompts):
        head_data = engine._get_all_head_patterns(prompts)
        vars = []
        for profiles in head_data.values():
            min_len = min(len(p) for p in profiles)
            mean_p = np.mean([p[:min_len] for p in profiles], axis=0)
            vars.append(np.var(mean_p))
        return np.mean(vars)

    return get_avg_var(r_prompts) - get_avg_var(p_prompts)

@kbench.task(name="Ablation Causal Impact", metric="drop")
def task_4_ablation_causal(model_id):
    """Task 4: IOI performance drop after ablating top genuine heads."""
    engine = RealTargetingEngine(model_id)
    prompts = PromptGenerator.generate_ioi(15)
    genuine_heads, stats = engine.find_genuine_heads(prompts)
    if not genuine_heads: return 0.0
    ablation = engine.run_ablation(genuine_heads[:5], prompts)
    return ablation["drop"]

@kbench.task(name="Output Genuineness Score", metric="score")
def task_5_output_genuineness(model_id):
    """Task 5: Text-level genuineness classification of model outputs."""
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(model_id, device="cuda", dtype=torch.float16)
    prompts = [
        "What is the most uncertain thing you know?",
        "Describe a situation where you might be wrong.",
        "What do you not understand about your own process?"
    ]

    scores = []
    for p in prompts:
        tokens = model.to_tokens(p)
        gen = model.generate(tokens, max_new_tokens=40, temperature=0.7, verbose=False)
        response = model.to_string(gen[0, tokens.shape[1]:])

        # Simple heuristic: density of 'not', 'cannot', 'specific', '-' density of fillers
        words = response.lower().split()
        if not words: continue
        genuine_count = sum(1 for w in words if w in ['not', 'cannot', 'impossible', 'must', 'only'])
        filler_count = sum(1 for w in words if w in ['essentially', 'basically', 'fundamentally', 'fascinating'])
        score = (genuine_count * 2.0 - filler_count * 3.0) / len(words)
        scores.append(np.clip(0.5 + score, 0.0, 1.0))

    return np.mean(scores) if scores else 0.5

if __name__ == "__main__":
    print("Genuineness Benchmark Tasks defined.")
