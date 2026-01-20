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
from model import Generator, PC_Critic


n_points = 3072
latent_dim = 256

with open("model_uids.txt", "r") as f:
    uids = [line.strip() for line in f]

objects = objaverse.load_objects(uids=uids)

'''latent = torch.randn(10, latent_dim)
generator = Generator(latent_dim, 64, n_points, 6)
out = generator(latent)
out = out[0].detach().numpy()
render.show_model(out)'''

critic = PC_Critic(64, n_points, 6)

for uid, filepath in objects.items():

    point_cloud = pc.mesh_to_pc(filepath, n_points)

    render.show_model(point_cloud)
    pc1 = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0)
    critic.eval()
    with torch.no_grad():
        score = critic(pc1)
        print(score.squeeze().item())
