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

batch_size = 64

transform = transforms.Compose([

    transforms.ToTensor(),
])

dataset = ImageFolder(root='dataset_images', transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

image,_ = dataset[0]
image = image.detach().cpu().permute(1, 2, 0).numpy()

print(image.shape)
plt.imshow(image)
plt.show()
