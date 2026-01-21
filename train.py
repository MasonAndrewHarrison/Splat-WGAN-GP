import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms 
import random
import point_cloud as pc
import render
from model import Generator, PC_Critic, initialize_weight
import point_cloud_dataset as pcd
import torch.optim as optim
import os


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    n_points = 3072
    latent_dim = 256
    batch_size = 64
    epochs = 10

    transform = pcd.PointCloudNormalize()

    dataset = pcd.Dataset(
        "model_uids.txt", 3072, 
        transform=transform, 
        value_type=torch.float32
    )
    print(dataset.__len__())
    print(dataset[327].shape)

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    fixed_latent = torch.randn(10, latent_dim)

    generator = Generator(latent_dim, 64, n_points, 6)
    critic = PC_Critic(64, n_points, 6)

    if os.path.exists("Generator.pth"):
        generator.load_state_dict(torch.load("Generator.pth", map_location=device))
        generator.to(device)

    else:
        initialize_weight(generator)

    opt_critic = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_gen = optim.Adam(critic.parameters(), lr=4e-4, betas=(0.0, 0.9))

    for epoch in range(epochs):
        for idx, real in enumerate(loader):

            pc = real[1].detach().numpy()
            print(pc.shape)
            render.show_model(pc)

if __name__ == "__main__":
    main()