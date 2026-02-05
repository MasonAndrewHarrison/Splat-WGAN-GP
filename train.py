import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.autograd')
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
from utils import gradient_penalty
from torch.amp import autocast, GradScaler
import os


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = pcd.PointCloudNormalize()
    n_points = 512
    latent_dim = 64
    batch_size = 64
    critic_iterations = 5
    epochs = 1000
    lambda_GP = 10 
    features = 64
    learning_rate = 1e-4
    pc_dim = 6
    

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
        prefetch_factor=3,
        persistent_workers=True
    )

    fixed_latent = torch.randn(10, latent_dim, device=device)

    generator = Generator(latent_dim, features, n_points, pc_dim)
    critic = PC_Critic(features, pc_dim)

    if os.path.exists("Generator.pth"):
        generator.load_state_dict(torch.load("Generator.pth", map_location=device))

    else:   
        initialize_weight(generator)

    if os.path.exists("Critic.pth"):
        critic.load_state_dict(torch.load("Critic.pth", map_location=device))
    
    else:
        initialize_weight(critic)

    generator.to(device) 
    critic.to(device)

    opt_gen = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    scaler_generator = GradScaler(device.__str__())
    scaler_critic = GradScaler(device.__str__())

    for epoch in range(epochs):

        for idx, real in enumerate(loader):

            real = real.to(device).float()
            real = real.permute(0, 2, 1)
            current_batch_size,_,_ = real.shape

            for _ in range(critic_iterations):

                opt_critic.zero_grad()

                with autocast(device_type=device.__str__(), dtype=torch.float16):
                    latent_space = torch.randn(current_batch_size, latent_dim, device=device)
                    fake = generator(latent_space)

                    fake_score = critic(fake.detach())
                    real_score = critic(real) 

                gp = gradient_penalty(critic, real.float(), fake.float(), device=device)

                with autocast(device_type=device.__str__(), dtype=torch.float16):
                    loss_critic = (
                        -(torch.mean(real_score) - torch.mean(fake_score))
                        + (lambda_GP * gp)
                    )
                
                scaler_critic.scale(loss_critic).backward()
                scaler_critic.step(opt_critic)
                scaler_critic.update()

            opt_gen.zero_grad()
            with autocast(device_type=device.__str__(), dtype=torch.float16):
                latent_space = torch.randn(current_batch_size, latent_dim, device=device)
                fake_pc = generator(latent_space)

                score = critic(fake_pc)
                loss_gen = -torch.mean(score)

            scaler_generator.scale(loss_gen).backward()
            scaler_generator.step(opt_gen)
            scaler_generator.update()

            if epoch % 25 == 0 and idx == 0:
                print("Saved")
                torch.save(generator.state_dict(), "Generator.pth")
                torch.save(critic.state_dict(), "Critic.pth")

            if epoch % 1 == 0 and idx == 0:
                wasserstein_distance = torch.mean(real_score) - torch.mean(fake_score)
                print(f"Epoch: {epoch} || Gen: {loss_gen:.4f} || Critic: {loss_critic:.4f} || W-dist: {wasserstein_distance:.4f} || GP: {gp:.4f}")

            if epoch % 100 == 0 and idx == 0:
                render.show_model(fake_pc[0])



if __name__ == "__main__":
    main()