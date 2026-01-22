import objaverse
import numpy as np
import point_cloud as pc
import render
import point_cloud_dataset as pcd
import torch


transforms = pcd.PointCloudNormalize()

dataset = pcd.Dataset("point_cloud_dataset.npy", transforms)

tensor = dataset[5]


print(type(tensor))

