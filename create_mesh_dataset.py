import objaverse
import re
import point_cloud as pc

def ValidateModel(uid):

    try:
        objects = objaverse.load_objects(uids=[uid])

        print(object)

        if objects.shape[0] == 1:
            return False, "Empty"
        
        if objects.shape[1] == 6:
            return False, "Invalid Shape"

        print(uid)
        return True, "Valid"
    
    except Exception as e:
        return False, str(e)


uids = objaverse.load_uids()

search_subset = uids[:50000]
annotations = objaverse.load_annotations(search_subset)

CAR_KEYWORDS = {
    "car", "automobile", "sedan", "suv", "coupe",
    "hatchback", "convertible", "wagon", "van",
    "pickup", "truck", "jeep"
}

car_uids = []

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

            car_uids.append(uid)
            #TODO make this just save the point cloud as np
            print(uid)

        except Exception as e:

            print(f"Error loading {uid}: {e}")


with open("model_uids.txt", "w") as f:
    for uid in car_uids:
        f.write(f"{uid}\n")

objects = objaverse.load_objects(uids=car_uids)
