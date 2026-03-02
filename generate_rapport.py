import json
import numpy as np
from datetime import datetime

def generate():
    try:
        with open("/app/benchmark_results.json", "r") as f:
            data = json.load(f)

        model = data["model"]
        stats = data["full_stats"]

        all_vars = [s["var_h"] for s in stats.values()]
        threshold = np.percentile(all_vars, 85)

        genuine_heads = [k for k, s in stats.items() if s["var_h"] >= threshold and s["collapses"] >= 1]

        # Aggregate stats by layer
        layer_stats = {}
        for head_id, s in stats.items():
            l = int(head_id.split('.')[0])
            if l not in layer_stats:
                layer_stats[l] = {"var": [], "mean": [], "collapses": 0}
            layer_stats[l]["var"].append(s["var_h"])
            layer_stats[l]["mean"].append(s["mean_h"])
            layer_stats[l]["collapses"] += s["collapses"]

        report = {
            "title": f"Precision Targeting Report: {model}",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_heads_scanned": len(stats),
                "genuine_heads_found": len(genuine_heads),
                "adaptive_threshold_v_h": float(threshold),
                "max_var_h": float(np.max(all_vars))
            },
            "findings": [
                f"Scanned {len(stats)} attention heads across {max(layer_stats.keys())+1} layers.",
                f"Adaptive threshold set at {threshold:.6f} (p85 of variance distribution).",
                f"Identified {len(genuine_heads)} heads with both high variance and collapse events.",
                f"Top variance layers: {sorted(layer_stats, key=lambda x: np.mean(layer_stats[x]['var']), reverse=True)[:5]}."
            ],
            "recommendations": [
                "Ablate top heads in the late layers to verify causal role in IOI.",
                "Compare these signatures against pure induction tasks to ensure task-specificity.",
                "Use the Real-Time Monitor on model outputs to catch 'elaboration pull' early."
            ],
            "top_genuine_heads": sorted(genuine_heads, key=lambda x: stats[x]["var_h"], reverse=True)[:10]
        }

        with open("FINAL_RAPPORT.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"Rapport generated. Threshold: {threshold:.6f}, Genuine: {len(genuine_heads)}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate()
