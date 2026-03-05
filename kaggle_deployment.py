import os, json, subprocess
from precision_targeting_engine import KaggleFullPipeline

def deploy():
    # Credentials should be set in the environment before running this script
    if not os.environ.get("KAGGLE_API_TOKEN") or not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("Error: KAGGLE_API_TOKEN and HUGGING_FACE_HUB_TOKEN must be set in the environment.")
        # return # Optionally exit if tokens are missing

    pipeline = KaggleFullPipeline(username="hichambedrani")

    # Prepare for public deployment
    dataset_dir = pipeline.prepare_dataset(
        title="Genuineness Benchmark: Mechanistic Engine v3",
        is_private=False
    )
    notebook_dir = pipeline.prepare_notebook(
        title="Genuineness Benchmark: Probing Reasoning circuits in LLMs",
        is_private=False
    )

    print("--- Pushing Dataset ---")
    # Update if exists, else create
    result = subprocess.run(["kaggle", "datasets", "version", "-p", dataset_dir, "-m", "Public Release V3"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Dataset version update failed, attempting create/metadata update...")
        subprocess.run(["kaggle", "datasets", "create", "-p", dataset_dir, "--public"])

    print("--- Pushing Kernel ---")
    subprocess.run(["kaggle", "kernels", "push", "-p", notebook_dir])

if __name__ == "__main__":
    deploy()
