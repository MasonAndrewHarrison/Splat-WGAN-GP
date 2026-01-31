import objaverse
import numpy as np
import point_cloud as pc
import render
import point_cloud_dataset as pcd
import torch


import subprocess
result = subprocess.run([
    "bash", "-c", 
    "source ~/TripoSR-fixed/venv/bin/activate && "
    "cd ~/TripoSR-fixed && "
    "python run.py examples/chair.png --output-dir output/test"
], capture_output=True, text=True)

print("Success:", result.returncode == 0)
print("Mesh at: ~/TripoSR-fixed/output/test/0/mesh.obj")


