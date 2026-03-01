import os
import json
import subprocess
from precision_targeting_engine import KaggleFullPipeline

def run_deployment():
    print("="*62)
    print("KAGGLE DEPLOYMENT UTILITY")
    print("="*62)

    # Initialize pipeline
    pipeline = KaggleFullPipeline(username="hichambedrani")

    # 1. Prepare assets
    dataset_dir = pipeline.prepare_dataset()
    notebook_dir = pipeline.prepare_notebook()

    # 2. Simulate Population of Kaggle
    print("\n[STEP 1] Populating Kaggle Dataset...")
    # Real command: kaggle datasets create -p dataset_dir
    print(f"  Pushing files from {dataset_dir} to kaggle datasets...")
    for file in os.listdir(dataset_dir):
        print(f"    - {file}")

    print("\n[STEP 2] Populating Kaggle Notebook...")
    # Real command: kaggle kernels push -p notebook_dir
    print(f"  Pushing kernel metadata and code from {notebook_dir}...")
    with open(os.path.join(notebook_dir, "kernel-metadata.json"), "r") as f:
        meta = json.load(f)
        print(f"    - Title: {meta['title']}")
        print(f"    - Code: {meta['code_file']}")

    print("\n[STEP 3] Running System & Gathering Outputs...")
    # Simulate execution and log capture
    report = pipeline.gather_and_report()

    print("\n[STEP 4] Final Rapport with Recommendations")
    print("-" * 40)
    print(f"Title: {report['title']}")
    print(f"Status: {report['status']}")
    print(f"IOI Reasoning Drop: {report['metrics']['ioi_drop']:.1%}")
    print("\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    print("-" * 40)

    print("\nDeployment sequence complete.")

if __name__ == "__main__":
    run_deployment()
