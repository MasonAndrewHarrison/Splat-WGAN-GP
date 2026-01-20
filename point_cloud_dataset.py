import torch
import point_cloud as pc
import objaverse
import render

class Dataset():
    def __init__(self, uid_filepath, n_points, transform):

        self.n_points=n_points
        self.transform=transform

        with open(uid_filepath, "r") as f:
            uids = [line.strip() for line in f]

        self.objects = list(objaverse.load_objects(uids=uids).items())
    
    def __len__(self):

        return len(self.objects)

    def __getitem__(self, idx):

        uids, filepath = self.objects[idx]

        point_cloud = pc.mesh_to_pc(filepath, self.n_points)

        return self.transform(point_cloud)
