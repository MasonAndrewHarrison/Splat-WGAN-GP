import torch
import point_cloud as pc
import objaverse
import render

class Dataset():
    def __init__(self, uid_filepath, n_points, transform, value_type):

        self.n_points=n_points
        self.transform=transform
        self.value_type=value_type

        with open(uid_filepath, "r") as f:
            self.uids = [line.strip() for line in f]
    
    def __len__(self):

        return len(self.uids)

    def __getitem__(self, idx):

        uid = self.uids[idx]

        try:

            objects = objaverse.load_objects(uids=[uid])
        
            if uid not in objects:
                raise ValueError(f"Failed to get UID: {uid}")

            filepath = objects[uid]
            point_cloud = pc.mesh_to_pc(filepath, self.n_points)
            point_cloud = torch.from_numpy(point_cloud).to(self.value_type)

            return self.transform(point_cloud)

        except Exception as e:

            print(f"Error loading {uid}: {e}")
            return torch.randn(self.n_points, 6)


class PointCloudNormalize():
    def __call__(self, pc):

        #TODO normalize pc here
        return pc
