import os, json, subprocess

def deploy_training():
    # Credentials should be set in the environment before running this script
    if not os.environ.get("KAGGLE_API_TOKEN"):
        print("Warning: KAGGLE_API_TOKEN not set in the environment.")

    # Update Dataset
    data_dir = "kaggle_deploy/genuine_model_data"
    subprocess.run(["kaggle", "datasets", "version", "-p", data_dir, "-m", "Improved script discovery"])

    # Update Kernel
    kernel_dir = "kaggle_deploy/train_kernel"
    script = """
import os, subprocess, sys, glob
print("--- Initializing ---")
# Robust discovery for Kaggle input
input_dirs = glob.glob("/kaggle/input/**/train_genuine.py", recursive=True)
if input_dirs:
    for d in input_dirs:
        sys.path.append(os.path.dirname(d))
        print(f"Added to path: {os.path.dirname(d)}")

try:
    from train_genuine import train
    train()
except ImportError as e:
    print(f"Error importing train_genuine: {e}")
    print("Files in current directory:", os.listdir("."))
    print("Files in input directory:", os.listdir("/kaggle/input"))
"""
    with open(os.path.join(kernel_dir, "train_run.py"), "w") as f: f.write(script)
    subprocess.run(["kaggle", "kernels", "push", "-p", kernel_dir])

if __name__ == "__main__":
    deploy_training()
