import torch
import point_cloud as pc
import objaverse
import numpy as np
import render
import math


class Dataset():
    def __init__(self, filepath, transform):

        self.filepath = filepath
        self.transform = transform
    
    def __len__(self):

        point_cloud_dataset = np.load(self.filepath, mmap_mode="r")
        return len(point_cloud_dataset)

    def __getitem__(self, idx):

        point_cloud_dataset = np.load(self.filepath, mmap_mode="r")
        point_cloud = point_cloud_dataset[idx].copy()

        tensor_pc = torch.from_numpy(point_cloud)

        if self.transform:
            tensor_pc = self.transform(tensor_pc)

        return tensor_pc



class PointCloudNormalize():
    def __call__(self, pc):

        #TODO normalize pc here
        avg_x = pc[:, 0].mean()
        avg_y = pc[:, 1].mean()
        avg_z = pc[:, 2].mean()

        pc[:, 0] = pc[:, 0] - avg_x
        pc[:, 1] = pc[:, 1] - avg_y
        pc[:, 2] = pc[:, 2] - avg_z

        # d_max = max_i sqrt( (x'_i)^2 + (y'_i)^2 + (z'_i)^2 ) over all points i
        # Ensures farthest point lies on unit sphere surface (norm=1)
        max_i = torch.linalg.vector_norm(pc[:, :3], dim=1)
        max_d = max_i.max()

        return pc/max_d
