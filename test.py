import objaverse

print("Loading LVIS annotations...")
lvis_annotations = objaverse.load_lvis_annotations()

# Debug: Check the structure
print(f"Total entries: {len(lvis_annotations)}")

# Look at first entry to understand structure
first_uid = list(lvis_annotations.keys())[0]
first_anno = lvis_annotations[first_uid]

print(f"\nFirst UID: {first_uid}")
print(f"Type of annotation: {type(first_anno)}")
print(f"Annotation content: {first_anno}")

# Try to understand the structure
if isinstance(first_anno, list):
    print(f"It's a list with {len(first_anno)} items")
    if len(first_anno) > 0:
        print(f"First item type: {type(first_anno[0])}")
        print(f"First item: {first_anno[0]}")
elif isinstance(first_anno, dict):
    print(f"It's a dict with keys: {first_anno.keys()}")
elif isinstance(first_anno, str):
    print(f"It's a string: {first_anno}")