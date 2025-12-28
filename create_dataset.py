import objaverse


uids = objaverse.load_uids()
subset = uids[:5]

objects = objaverse.load_objects(uids=subset)


with open("model_uids.txt", "w") as f:
    for uid in subset:
        f.write(f"{uid}\n")


