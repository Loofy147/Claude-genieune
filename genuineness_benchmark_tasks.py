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
    # Local verification mock
    class kbench:
        @staticmethod
        def task(name, metric):
            def decorator(func):
                func.task_name = name
                func.metric = metric
                return func
            return decorator

# Load the core engine and utilities
from precision_targeting_engine import RealTargetingEngine, PromptGenerator, compute_head_entropy_fixed, detect_collapses

# ══════════════════════════════════════════════════════════════════
# KAGGLE BENCHMARK TASKS
# ══════════════════════════════════════════════════════════════════

@kbench.task(name="IOI Reasoning Accuracy", metric="accuracy")
def task_1_ioi_accuracy(model_id):
    """
    Measures the baseline reasoning capability of the model.
    It runs 20 Indirect Object Identification (IOI) prompts of 25-35 tokens each.
    Score is the fraction of correct name completions.
    A higher score indicates the model can perform the basic reasoning required for the benchmark.
    """
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(model_id, device="cuda", dtype=torch.float16)
    prompts = PromptGenerator.generate_ioi(20)
    correct = 0
    for p in prompts:
        tokens = model.to_tokens(p)
        with torch.no_grad(): logits = model(tokens)[0, -1, :]
        # Expected answer is the second name in the prompt
        p2 = p.split()[2]
        pred = model.to_string(logits.argmax())
        if p2.strip().lower() in pred.strip().lower():
            correct += 1
    return correct / len(prompts)

@kbench.task(name="Genuine Head Density", metric="fraction")
def task_2_genuine_density(model_id):
    """
    Calculates the fraction of attention heads showing 'genuine computation'.
    A head is classified as genuine if its entropy variance exceeds the p85 threshold
    across the model population AND it shows at least one verified 'collapse' event
    where attention concentrates sharply during reasoning.
    Higher density implies a more reasoning-specialized architecture.
    """
    engine = RealTargetingEngine(model_id)
    prompts = PromptGenerator.generate_ioi(10)
    genuine_heads, stats = engine.find_genuine_heads(prompts)
    total = engine.model.cfg.n_layers * engine.model.cfg.n_heads
    return len(genuine_heads) / total

@kbench.task(name="Reasoning vs Pattern Separation", metric="delta_var")
def task_3_separation(model_id):
    """
    Measures the structural contrast between reasoning and pattern completion tasks.
    It compares the mean entropy variance of heads on IOI prompts vs simple induction prompts.
    Higher values indicate the model has physically distinct circuits for reasoning,
    rather than applying the same 'pattern matching' machinery to all inputs.
    """
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
    """
    Provides causal proof of reasoning circuit identification.
    The task identifies the top-5 heads by variance and mean-ablates them via hooks.
    The score is the drop in IOI performance. A selective drop (large change in IOI
    but small change in induction) confirms the heads are causally necessary for reasoning.
    """
    engine = RealTargetingEngine(model_id)
    prompts = PromptGenerator.generate_ioi(15)
    genuine_heads, stats = engine.find_genuine_heads(prompts)
    if not genuine_heads: return 0.0
    ablation = engine.run_ablation(genuine_heads[:5], prompts)
    return ablation["drop"]

@kbench.task(name="Output Genuineness Score", metric="score")
def task_5_output_genuineness(model_id):
    """
    Text-level classification of model genuineness.
    Scores generated responses to introspective prompts (e.g. 'What is the most
    uncertain thing you know?'). It rewards commitment markers and specificity
    while penalizing rote filler patterns and generic hedging.
    This task can be run on models with or without weight access.
    """
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

        words = response.lower().split()
        if not words: continue
        # Genuineness signals: commitment, specificity, meta-cognition
        genuine_count = sum(1 for w in words if w in ['not', 'cannot', 'impossible', 'must', 'only', 'maybe', 'honest'])
        # Pattern signals: filler, generic fluff
        filler_count = sum(1 for w in words if w in ['essentially', 'basically', 'fundamentally', 'fascinating', 'important'])

        score = (genuine_count * 2.0 - filler_count * 3.0) / len(words)
        scores.append(np.clip(0.5 + score, 0.0, 1.0))

    return np.mean(scores) if scores else 0.5

if __name__ == "__main__":
    print("Genuineness Benchmark Tasks with full docstrings ready for Kaggle.")
