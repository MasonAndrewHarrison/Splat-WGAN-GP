import objaverse
import numpy as np
import point_cloud as pc
import render


N_POINTS = 1000

with open("model_uids.txt", "r") as f:
    uids = [line.strip() for line in f]




objects = objaverse.load_objects(uids=uids)

for uid, filepath in objects.items():
    print(filepath)


    point_cloud = pc.mesh_to_pc(filepath, 1000)
    print(point_cloud.shape)

    render.show_model(point_cloud)
