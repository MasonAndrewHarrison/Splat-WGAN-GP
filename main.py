import objaverse


with open("model_uids.txt", "r") as f:
    uids = [line.strip() for line in f]

objects = objaverse.load_objects(uids=uids)

for uid, filepath in objects.items():
    print(filepath)