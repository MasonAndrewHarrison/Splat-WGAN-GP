import torch
import point_cloud as pc
import objaverse
import numpy as np
import render

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
        return pc
