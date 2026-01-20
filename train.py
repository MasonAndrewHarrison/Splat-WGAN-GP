import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import objaverse
import numpy as np
import point_cloud as pc
import render


N_POINTS = 1000

with open("model_uids.txt", "r") as f:
    uids = [line.strip() for line in f]

objects = objaverse.load_objects(uids=uids)

for uid, filepath in objects.items():
    print(filepath)


    point_cloud = pc.mesh_to_pc(filepath, 3072)
    print(point_cloud.shape)
    print(point_cloud)

    render.show_model(point_cloud)
