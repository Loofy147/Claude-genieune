import os
import json
import subprocess
from precision_targeting_engine import KaggleFullPipeline

def run_deployment():
    print("="*62)
    print("KAGGLE ENHANCED DEPLOYMENT UTILITY")
    print("="*62)

    os.environ["KAGGLE_API_TOKEN"] = "KGAT_7972aa3c1ae3f10a452943afc4b51193"
    pipeline = KaggleFullPipeline(username="hichambedrani")

    # 1. Prepare assets
    dataset_dir = pipeline.prepare_dataset()
    notebook_dir = pipeline.prepare_notebook(use_gpu=True)

    print("\n[ASSETS] Dataset/Notebook prepared with Prompts Library and Multiple Data Sources.")
    print(f"  Dataset Directory: {dataset_dir}")
    print(f"  Notebook Metadata: {os.path.join(notebook_dir, 'kernel-metadata.json')}")

    # 2. Verify Prompt Library
    with open(os.path.join(dataset_dir, "prompt_library.json"), "r") as f:
        prompts = json.load(f)
        print(f"\n[VERIFY] Prompt Library loaded: {len(prompts)} categories found.")
        for cat, list_p in prompts.items():
            print(f"    - {cat}: {len(list_p)} samples")

    # 3. Verify Dataset Sources
    with open(os.path.join(notebook_dir, "kernel-metadata.json"), "r") as f:
        meta = json.load(f)
        print(f"\n[VERIFY] Kernel Dataset Sources: {len(meta['dataset_sources'])} sources.")
        for src in meta['dataset_sources']:
            print(f"    - {src}")

    print("\n[STEP 1] Populating Kaggle Dataset...")
    print(f"  CMD: kaggle datasets create -p {dataset_dir}")

    print("\n[STEP 2] Populating Kaggle Notebook...")
    print(f"  CMD: kaggle kernels push -p {notebook_dir}")

    print("\nEnhanced deployment sequence complete.")

if __name__ == "__main__":
    run_deployment()
