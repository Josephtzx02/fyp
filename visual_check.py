import cv2
from pathlib import Path
import yaml
import random
import math
import numpy as np

# ======================
# CONFIG
# ======================
DATA_YAML = "data.yaml"       # your data.yaml
SET = "test"                 # choose "train", "val", or "test"
SHOW_WINDOW = True            # True = display image, False = save images
PAUSE_BETWEEN = True          # True = wait for Enter to show next image
SAVE_PATH = "visual_check_output"  # only used if SHOW_WINDOW=False
SAMPLE_SIZE = 10              # number of images to check (None = all)
LABEL_FOLDER_NAME = "labels"  
# ======================

# Load data.yaml
with open(DATA_YAML, "r") as f:
    data = yaml.safe_load(f)

base_path = Path(data.get("path", "."))

image_folder = base_path / data[SET]
all_images = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))

if SAMPLE_SIZE and SAMPLE_SIZE < len(all_images):
    image_paths = random.sample(all_images, SAMPLE_SIZE)
else:
    image_paths = all_images

print(f"[INFO] Total images selected for {SET}: {len(image_paths)}")

Path(SAVE_PATH).mkdir(exist_ok=True)
class_names = data.get("names", {})
label_folder = base_path / LABEL_FOLDER_NAME / SET

for idx, img_path in enumerate(image_paths):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARNING] Could not read image: {img_path}")
        continue

    # Match any label containing image stem
    matching_labels = list(label_folder.glob(f"*{img_path.stem}*.txt"))
    if matching_labels:
        label_file = matching_labels[0]
        with open(label_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            # OBB 5 numbers
            if len(coords) == 5:
                cx, cy, w, h, angle = coords
                cx *= img.shape[1]
                cy *= img.shape[0]
                w  *= img.shape[1]
                h  *= img.shape[0]
                angle_deg = angle * 180 / math.pi
                rect = ((cx, cy), (w, h), angle_deg)
                box = cv2.boxPoints(rect)
                box = box.astype(int)
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                cv2.putText(img, class_names.get(cls_id, str(cls_id)), (int(cx), int(cy)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Normal bbox 4 numbers
            elif len(coords) == 4:
                x1, y1, x2, y2 = map(int, coords)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, class_names.get(cls_id, str(cls_id)), (x1, max(y1-10,0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Polygon 8 numbers
            elif len(coords) == 8:
                pts = []
                for i in range(0,8,2):
                    x = int(coords[i]*img.shape[1])
                    y = int(coords[i+1]*img.shape[0])
                    pts.append([x,y])
                pts = np.array(pts, dtype=int).reshape((-1,1,2))
                cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
                # put class name near first point
                cv2.putText(img, class_names.get(cls_id, str(cls_id)), (pts[0][0][0], max(pts[0][0][1]-10,0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                print(f"[WARNING] Unexpected label format: {label_file} -> {coords}")
    else:
        print(f"[WARNING] No label found for image: {img_path.name}")

    if SHOW_WINDOW:
        cv2.imshow(f"Image {idx+1}/{len(image_paths)}", img)
        if PAUSE_BETWEEN:
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        save_file = Path(SAVE_PATH) / img_path.name
        cv2.imwrite(str(save_file), img)

print("[INFO] Visual check complete!")
