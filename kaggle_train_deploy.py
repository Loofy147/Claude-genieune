import os, json, subprocess

def deploy_training():
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_7972aa3c1ae3f10a452943afc4b51193"

    # Update Dataset
    data_dir = "kaggle_deploy/genuine_model_data"
    subprocess.run(["kaggle", "datasets", "version", "-p", data_dir, "-m", "Path fix"])

    # Update Kernel
    kernel_dir = "kaggle_deploy/train_kernel"
    script = """
import os, subprocess, sys, glob
print("--- Initializing ---")
files = glob.glob("/kaggle/input/**/train_genuine.py", recursive=True)
if files: sys.path.append(os.path.dirname(files[0]))

from train_genuine import train
train()
"""
    with open(os.path.join(kernel_dir, "train_run.py"), "w") as f: f.write(script)
    subprocess.run(["kaggle", "kernels", "push", "-p", kernel_dir])

if __name__ == "__main__":
    deploy_training()
