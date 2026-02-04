import objaverse
import re
import point_cloud as pc
import numpy as np

n_points = 512

uids = objaverse.load_uids()
search_subset = uids[:90000]
annotations = objaverse.load_annotations(search_subset)

CAR_KEYWORDS = {
    "car", "automobile", "sedan", "suv", "coupe",
    "hatchback", "convertible", "wagon", "van",
    "pickup", "truck", "jeep"
}

point_cloud_dataset = []

usable_idx = 0
for idx, (uid, anno) in enumerate(annotations.items()):
    name = anno.get('name', '').lower()

    tokens = set(re.findall(r"[a-z]+", name.lower()))
    
    if any(w in tokens for w in CAR_KEYWORDS):

        try:

            objects = objaverse.load_objects(uids=[uid])
        
            if uid not in objects:
                raise ValueError(f"Failed to get UID: {uid}")

            filepath = objects[uid]
            point_cloud = pc.mesh_to_pc(filepath, n_points)

            usable_idx += 1
            point_cloud_dataset.append(point_cloud.astype(np.float16))
            print(f"{usable_idx} || {idx} / {len(annotations.items())} || {(100* idx / len(annotations.items())):.2f}% || {uid}")

        except Exception as e:

            print(f"Error loading {uid}: {e}")


np.save("point_cloud_dataset.npy", point_cloud_dataset)

