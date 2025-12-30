import objaverse
import trimesh
import render


with open("model_uids.txt", "r") as f:
    uids = [line.strip() for line in f]

objects = objaverse.load_objects(uids=uids)

for uid, filepath in objects.items():
    print(filepath)


    mesh = trimesh.load(filepath, force='mesh')

    vertices = mesh.vertices

    print(vertices)
    render.show_model(vertices)
