import objaverse
import re


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
        print(name)
        car_uids.append(uid)

with open("model_uids.txt", "w") as f:
    for uid in car_uids:
        f.write(f"{uid}\n")

objects = objaverse.load_objects(uids=car_uids)
