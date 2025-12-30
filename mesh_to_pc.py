import trimesh
import numpy as np


def mesh_to_pc(filepath, n_points):

    mesh = trimesh.load(filepath, force='mesh')

    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / mesh.scale)

    mesh.remove_unreferenced_vertices()

    points, face_index = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_index]


    point_cloud = np.concatenate([points, normals], axis=1)

    return point_cloud