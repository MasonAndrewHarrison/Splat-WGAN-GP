import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import os
import shutil

os.makedirs("temp/all", exist_ok=True)

transform = transforms.Compose([transforms.ToTensor(),])

dataset = ImageFolder(root='dataset_images', transform=transform)

image = 0
true_image_index = 0

for i, (image,_) in enumerate(dataset):

    if image.shape == (3, 240, 320):

        first_pixel = image[:, 0, 0]
        last_pixel = image[:, 239, 319]
        target = torch.tensor([1.0, 1.0, 1.0])

        if torch.equal(first_pixel, target) and torch.equal(last_pixel, target):

            true_image_index += 1

            print(image.shape)
            save_image(image, f"temp/all/test{i}.png")

            exit()



shutil.rmtree("dataset_images/")
