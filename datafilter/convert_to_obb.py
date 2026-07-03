import json
import numpy as np
import cv2
from rdp import rdp

INPUT_JSON = "annotations.coco.json"
STEP1_JSON = "annotations_rdp.json"
STEP2_JSON = "annotations_obb.json"
EPSILON = 3  # RDP simplification

def order_clockwise(pts):
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
    return pts[np.argsort(angles)]

# Load annotations
with open(INPUT_JSON, "r") as f:
    coco = json.load(f)

# ================= STEP 1: RDP Simplify =================
for ann in coco["annotations"]:
    if "segmentation" not in ann or not ann["segmentation"]:
        continue

    seg = ann["segmentation"][0]
    pts = np.array(seg).reshape(-1, 2).astype(np.float32)

    if len(pts) < 4:
        continue

    simplified = rdp(pts.tolist(), epsilon=EPSILON)
    ann["segmentation"] = [np.array(simplified, dtype=np.float32).flatten().tolist()]

# Save RDP results for visual check
with open(STEP1_JSON, "w") as f:
    json.dump(coco, f, indent=2)
print("STEP 1 DONE → RDP simplified JSON saved")

# ================= STEP 2: OpenCV Filter & OBB =================
for ann in coco["annotations"]:
    if "segmentation" not in ann or not ann["segmentation"]:
        continue

    seg = ann["segmentation"][0]
    pts = np.array(seg).reshape(-1, 2).astype(np.float32)

    if len(pts) < 4:
        continue

    # Convex Hull
    hull = cv2.convexHull(pts)

    if cv2.contourArea(hull) < 50:
        continue

    # Oriented Bounding Box
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = order_clockwise(box)

    ann["segmentation"] = [box.flatten().tolist()]

# Save final OBB results
with open(STEP2_JSON, "w") as f:
    json.dump(coco, f, indent=2)
print("STEP 2 DONE → OBB JSON saved")
