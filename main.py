import objaverse
import numpy as np
import point_cloud as pc
import render
import point_cloud_dataset as pcd
import torch


#TODO get this to use TriplaneGausian instead
#TODO make or find a gausian render
#TODO make a gausian render in c/c++

import subprocess
result = subprocess.run([
    "bash", "-c", 
    "source ~/TripoSR-fixed/venv/bin/activate && "
    "cd ~/TripoSR-fixed && "
    "python run.py examples/chair.png --output-dir output/test"
], capture_output=True, text=True)

print("Success:", result.returncode == 0)
print("Mesh at: ~/TripoSR-fixed/output/test/0/mesh.obj")


