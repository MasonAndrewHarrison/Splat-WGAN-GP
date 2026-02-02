import open3d as o3d
import torch
import numpy as np

def show_model(point_cloud):

    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().numpy()
        point_cloud = np.transpose(point_cloud, (1, 0))
        print(point_cloud.shape)

    if point_cloud is None:
        raise ValueError("Array is empty") 

    else:
        pcd_list = o3d.geometry.PointCloud()
        print(point_cloud.shape)
        pcd_list.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        _, dim_size = point_cloud.shape
        if dim_size > 3:
            pcd_list.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])

        #o3d.visualization.draw_geometries([pcd_list])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_list)
    
        render_option = vis.get_render_option()
        render_option.point_size = 30.0
        
        vis.run()
        vis.destroy_window()