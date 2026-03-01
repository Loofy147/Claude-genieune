
# TransformerLens measurement — run this with model access
# Tests the dynamic entropy genuineness hypothesis on real circuits

import transformer_lens
import torch
import numpy as np
from collections import defaultdict

def measure_attention_entropy(model, prompts: list, layer_range=None):
    """
    Measure dynamic entropy genuineness for each attention head
    across a set of prompts.
    
    Returns: dict of head -> dynamic_genuineness_score
    """
    if layer_range is None:
        layer_range = range(model.cfg.n_layers)
    
    head_entropy_profiles = defaultdict(list)
    
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            tokens = model.to_tokens(prompt)
            _, cache = model.run_with_cache(tokens)
            
            for layer in layer_range:
                # Shape: (batch, heads, seq, seq)
                attn_weights = cache[f"blocks.{layer}.attn.hook_pattern"]
                
                for head in range(model.cfg.n_heads):
                    weights = attn_weights[0, head].cpu().numpy()  # (seq, seq)
                    
                    # Compute entropy per query position
                    entropies = []
                    for row in weights:
                        row = np.maximum(row, 1e-10)
                        h = -np.sum(row * np.log2(row))
                        max_h = np.log2(len(row))
                        entropies.append(h / max_h)
                    
                    head_entropy_profiles[(layer, head)].append(entropies)
    
    # Compute dynamic genuineness for each head
    results = {}
    for (layer, head), profiles in head_entropy_profiles.items():
        # Average entropy profile across prompts
        mean_profile = np.mean(profiles, axis=0)
        
        # Dynamic metrics
        std_h = float(np.std(mean_profile))
        range_h = float(np.max(mean_profile) - np.min(mean_profile))
        
        # Find collapse points
        deltas = np.diff(mean_profile)
        collapses = int(np.sum(deltas < -0.20))
        
        # Score
        score = (
            min(std_h / 0.35, 1.0) * 0.40 +
            min(range_h / 0.80, 1.0) * 0.35 +
            min(collapses * 0.40, 1.0) * 0.25
        )
        
        # Penalty for induction-like static low entropy
        if np.mean(mean_profile) < 0.20 and std_h < 0.05:
            score *= 0.2
        
        results[(layer, head)] = {
            "dynamic_genuineness": round(float(score), 3),
            "mean_entropy": round(float(np.mean(mean_profile)), 3),
            "std_entropy": round(std_h, 3),
            "collapse_count": collapses,
            "classification": (
                "GENUINE_COMPUTATION" if score > 0.55
                else "PATTERN_COMPLETION" if score < 0.35
                else "UNCERTAIN"
            )
        }
    
    return results


def compare_reasoning_vs_pattern(model):
    """
    The key experiment: compare entropy dynamics on
    genuine reasoning tasks vs pure pattern completion tasks.
    """
    # Genuine reasoning prompts (IOI-style)
    reasoning_prompts = [
        "John and Mary went to the store. John gave the apple to",
        "The cat sat on the mat. The dog sat on the",
        "Alice told Bob that she would meet him. Bob waited for",
    ]
    
    # Pure pattern completion prompts (induction)
    pattern_prompts = [
        "A B C A B C A B",
        "1 2 3 1 2 3 1 2",
        "red blue green red blue green red",
    ]
    
    reasoning_scores = measure_attention_entropy(model, reasoning_prompts)
    pattern_scores = measure_attention_entropy(model, pattern_prompts)
    
    # Find heads that differentiate
    differentiating_heads = []
    for head_key in reasoning_scores:
        if head_key in pattern_scores:
            r_score = reasoning_scores[head_key]["dynamic_genuineness"]
            p_score = pattern_scores[head_key]["dynamic_genuineness"]
            separation = r_score - p_score
            if abs(separation) > 0.2:
                differentiating_heads.append({
                    "head": head_key,
                    "reasoning_score": r_score,
                    "pattern_score": p_score,
                    "separation": round(separation, 3)
                })
    
    differentiating_heads.sort(key=lambda x: abs(x["separation"]), reverse=True)
    return differentiating_heads
