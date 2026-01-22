import objaverse
import re
import point_cloud as pc
import numpy as np


uids = objaverse.load_uids()

search_subset = uids[:5000]
annotations = objaverse.load_annotations(search_subset)

CAR_KEYWORDS = {
    "car", "automobile", "sedan", "suv", "coupe",
    "hatchback", "convertible", "wagon", "van",
    "pickup", "truck", "jeep"
}

point_cloud_dataset = []

for uid, anno in annotations.items():
    name = anno.get('name', '').lower()

    tokens = set(re.findall(r"[a-z]+", name.lower()))
    
    if any(w in tokens for w in CAR_KEYWORDS):

        try:

            objects = objaverse.load_objects(uids=[uid])
        
            if uid not in objects:
                raise ValueError(f"Failed to get UID: {uid}")

            filepath = objects[uid]
            point_cloud = pc.mesh_to_pc(filepath, 3000)

            point_cloud_dataset.append(point_cloud)
            print(uid)

        except Exception as e:

            print(f"Error loading {uid}: {e}")


np.save("point_cloud_dataset.npy", point_cloud_dataset)

