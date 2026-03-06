
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
