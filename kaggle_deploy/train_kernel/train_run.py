
import os, subprocess, sys, glob
print("--- Initializing ---")
files = glob.glob("/kaggle/input/**/train_genuine.py", recursive=True)
if files: sys.path.append(os.path.dirname(files[0]))

from train_genuine import train
train()
