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
    #device = "cpu"

    n_points = 3072
    latent_dim = 64
    batch_size = 64
    critic_iterations = 5
    epochs = 1000
    weight_clip = 0.01
    features = 32
    pc_dim = 6
    transform = pcd.PointCloudNormalize()

    dataset = pcd.Dataset(
        "point_cloud_dataset.npy",
        transform=transform, 
    )

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    fixed_latent = torch.randn(10, latent_dim).to(device)

    generator = Generator(latent_dim, features, n_points, pc_dim)
    critic = PC_Critic(features, pc_dim)

    if os.path.exists("Generator.pth"):
        generator.load_state_dict(torch.load("Generator.pth", map_location=device))
        generator.to(device)

    else:   
        initialize_weight(generator)

    if os.path.exists("Critic.pth"):
        critic.load_state_dict(torch.load("Critic.pth", map_location=device))
        critic.to(device)
    
    else:
        initialize_weight(critic)

    generator.to(device) 
    critic.to(device)

    opt_gen = optim.Adam(generator.parameters(), lr=4e-4, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))

    for epoch in range(epochs):

        for idx, real in enumerate(loader):

            real = real.to(device).float()
            real = real.permute(0, 2, 1)
            current_batch_size,_,_ = real.shape

            for _ in range(critic_iterations):

                latent_space = torch.randn(current_batch_size, latent_dim).to(device)
                fake = generator(latent_space)

                fake_score = critic(fake.detach())
                real_score = critic(real) 

                loss_critic = -(torch.mean(real_score)-torch.mean(fake_score))
                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                for p in critic.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)

            latent_space = torch.randn(current_batch_size, latent_dim).to(device)
            fake_pc = generator(latent_space)

            score = critic(fake_pc)
            loss_gen = -torch.mean(score)

            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if epoch % 1 == 0 and idx == 0:
                print(f"Epoch: {epoch} of {epochs} || print gen loss: {loss_gen:.4f} || print critic loss: {loss_critic:.4f}")

            if epoch % 1 == 0 and idx == 0:
                print("Saved")
                torch.save(generator.state_dict(), "Generator.pth")
                torch.save(critic.state_dict(), "Critic.pth")

            if epoch % 50 == 0 and idx == 0:
                render.show_model(fake_pc[0])



if __name__ == "__main__":
    main()