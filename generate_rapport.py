import json
import numpy as np
from datetime import datetime

def generate():
    try:
        with open("/app/benchmark_results.json", "r") as f:
            data = json.load(f)

        model = data["model"]
        genuine_heads = data["genuine_heads"]
        stats = data["full_stats"]

        # Aggregate stats by layer
        layer_stats = {}
        for head_id, s in stats.items():
            l = int(head_id.split('.')[0])
            if l not in layer_stats:
                layer_stats[l] = {"var": [], "mean": [], "collapses": 0}
            layer_stats[l]["var"].append(s["var_h"])
            layer_stats[l]["mean"].append(s["mean_h"])
            layer_stats[l]["collapses"] += s["collapses"]

        # Final Report
        report = {
            "title": f"Precision Targeting Report: {model}",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_heads_scanned": len(stats),
                "genuine_heads_found": len(genuine_heads),
                "target_range_scanned": "Layers 21-32 (and beyond)"
            },
            "findings": [
                f"Scanned {len(stats)} attention heads across {max(layer_stats.keys())+1} layers.",
                f"Identified {len(genuine_heads)} reasoning circuits meeting GENUINE_DIFFUSE criteria.",
                f"Observed highest average entropy variance in layers {sorted(layer_stats, key=lambda x: np.mean(layer_stats[x]['var']), reverse=True)[:3]}."
            ],
            "recommendations": [
                "Target identified heads in layers 21-32 for mean ablation to confirm causal reasoning necessity.",
                "Protect late-layer heads with high variance (>0.12) to maintain response genuineness.",
                "Deploy the Real-Time Monitor with a 'STOP' threshold of 0.4 on detected elaboration pulls."
            ],
            "top_genuine_heads": genuine_heads[:10]
        }

        with open("FINAL_RAPPORT.json", "w") as f:
            json.dump(report, f, indent=2)

        print("Final Rapport generated successfully.")
    except Exception as e:
        print(f"Error generating rapport: {e}")

if __name__ == "__main__":
    generate()
