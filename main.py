import objaverse
import numpy as np
import point_cloud as pc
import render


N_POINTS = 1000


point_cloud_dataset = np.load("point_cloud_dataset.npy")
print(type(point_cloud_dataset))

for point_cloud in point_cloud_dataset:

    render.show_model(point_cloud)


