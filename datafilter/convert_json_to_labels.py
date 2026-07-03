import json
import os

JSON_FILE = "annotations.coco.json"
LABEL_ROOT = r"C:\Users\Joseph\Downloads\yolov8_obb_project\book_obb_dataset\labels\test"

os.makedirs(LABEL_ROOT, exist_ok=True)

with open(JSON_FILE, "r") as f:
    coco = json.load(f)

image_map = {img["id"]: img for img in coco["images"]}

for ann in coco["annotations"]:

    seg = ann.get("segmentation")

    # Skip masks (RLE)
    if isinstance(seg, dict):
        continue

    # Skip empty or broken polygons
    if not seg or len(seg[0]) < 8:
        continue

    img = image_map[ann["image_id"]]
    w, h = img["width"], img["height"]
    name = os.path.splitext(os.path.basename(img["file_name"]))[0]

    poly = seg[0]

    pts = []
    for i in range(0, 8, 2):
        pts.append(f"{poly[i]/w:.6f} {poly[i+1]/h:.6f}")

    label_path = os.path.join(LABEL_ROOT, name + ".txt")
    class_id = ann["category_id"] - 1

    with open(label_path, "a") as f:
        f.write(f"{class_id} " + " ".join(pts) + "\n")

print("✅ YOLOv8-OBB label files created")
