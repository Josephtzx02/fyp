import json
import cv2
import numpy as np
import os
from collections import defaultdict

IMG_DIR = "."  # path to your images
JSON_ORIG = "annotations.coco.json"
JSON_RDP = "annotations_rdp.json"

# Load JSON files
with open(JSON_ORIG, "r") as f:
    coco_orig = json.load(f)

with open(JSON_RDP, "r") as f:
    coco_rdp = json.load(f)

# Build image_id → filename map
id_to_name = {img["id"]: img["file_name"] for img in coco_orig["images"]}

# Group annotations by image
ann_orig = defaultdict(list)
ann_rdp = defaultdict(list)

for a in coco_orig["annotations"]:
    ann_orig[a["image_id"]].append(a)

for a in coco_rdp["annotations"]:
    ann_rdp[a["image_id"]].append(a)

for image_id, filename in id_to_name.items():
    img_path = os.path.join(IMG_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Draw ORIGINAL polygons (RED)
    for a in ann_orig[image_id]:
        if not a.get("segmentation"):
            continue
        seg = a["segmentation"][0]
        if len(seg) < 6:
            continue
        pts = np.array(seg).reshape(-1, 2).astype(int)
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)

    # Draw RDP-simplified polygons (GREEN)
    for a in ann_rdp[image_id]:
        if not a.get("segmentation"):
            continue
        seg = a["segmentation"][0]
        if len(seg) < 4:
            continue  # skip broken polygons
        pts = np.array(seg).reshape(-1, 2).astype(int)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)

    # Resize for display if needed
    scale = 1  # try 2.0–3.0 if small images
    h, w = img.shape[:2]
    img_show = cv2.resize(img, (int(w*scale), int(h*scale)))

    cv2.imshow("RED = original | GREEN = RDP", img_show)
    key = cv2.waitKey(0)
    if key == 27:  # ESC to quit early
        break

cv2.destroyAllWindows()
