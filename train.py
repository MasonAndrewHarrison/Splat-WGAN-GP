import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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
#loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


correct_dim = False
image = 0

while not correct_dim:

    random_int = random.randint(1, 60000)

    temp_image, label = dataset[random_int]



    if temp_image.shape == (3, 240, 320):

        first_pixel = temp_image[:, 0, 0]
        last_pixel = temp_image[:, 239, 319]
        target = torch.tensor([1.0, 1.0, 1.0])

        if torch.equal(first_pixel, target) and torch.equal(last_pixel, target):

            image = temp_image
            correct_dim = True
            #image = image.permute(1, 2, 0)
            #image = image * 0.5 + 0.5

            print(image.shape)
            save_image(image, "test.png")

'''
print(image.shape)
plt.imshow(image)
plt.show()'''
