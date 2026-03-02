import os, json, subprocess
from precision_targeting_engine import KaggleFullPipeline

def deploy():
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_7972aa3c1ae3f10a452943afc4b51193"
    pipeline = KaggleFullPipeline(username="hichambedrani")

    dataset_dir = pipeline.prepare_dataset()
    notebook_dir = pipeline.prepare_notebook(model_name="gpt2-xl")

    print("--- Pushing Dataset ---")
    subprocess.run(["kaggle", "datasets", "version", "-p", dataset_dir, "-m", "V3 Complete"])

    print("--- Pushing Kernel ---")
    subprocess.run(["kaggle", "kernels", "push", "-p", notebook_dir])

if __name__ == "__main__":
    deploy()
