import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
from datetime import datetime

BASE_DIR = "D435_HD/"
RGB_DIR = os.path.join(BASE_DIR, "RGB")
DEPTH_RAW_DIR = os.path.join(BASE_DIR, "DEPTH_RAW")
DEPTH_CM_DIR = os.path.join(BASE_DIR, "DEPTH_COLORMAP")
META_DIR = os.path.join(BASE_DIR, "META")
INTR_DIR = os.path.join(BASE_DIR, "INTRINSICS")

for d in [RGB_DIR, DEPTH_RAW_DIR, DEPTH_CM_DIR, META_DIR, INTR_DIR]:
    os.makedirs(d, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) # RGB: 1280x720 (HD) -> Much sharper for YOLO
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30) # Depth: 848x480 -> Native D435 resolution, good match for wide RGB

align_to = rs.stream.color
align = rs.align(align_to) # Align Depth to Color (Depth will be upscaled to 1280x720 automatically)

profile = pipeline.start(config)

color_stream = profile.get_stream(rs.stream.color)
color_intr = color_stream.as_video_stream_profile().get_intrinsics() #Intrinsics change with resolution

intr_dict = {
    "width": color_intr.width,
    "height": color_intr.height,
    "fx": color_intr.fx,
    "fy": color_intr.fy,
    "ppx": color_intr.ppx,
    "ppy": color_intr.ppy,
    "model": str(color_intr.model),
    "coeffs": color_intr.coeffs
}

with open(os.path.join(INTR_DIR, "camera_intrinsics.json"), "w") as f:
    json.dump(intr_dict, f, indent=4)

print(f"Intrinsics saved for {color_intr.width}x{color_intr.height} resolution.")

print("Press SPACE to capture, press q to quit.")

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    rgb_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not rgb_frame or not depth_frame:
        continue

    rgb = np.asanyarray(rgb_frame.get_data())
    depth_raw = np.asanyarray(depth_frame.get_data())

    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_raw, alpha=0.08), #(0.08 for better close-up contrast)
        cv2.COLORMAP_JET
    )
    full_res_preview = np.hstack((rgb, depth_colormap)) #Stack images side-by-side. Total width = 1280 + 1280 = 2560 (Too big!)

    preview = cv2.resize(full_res_preview, None, fx=1, fy=1) # Resize just for display. Scale down by 50% (0.5). Result size: 1280x360 (Fits easily on laptop)

    cv2.imshow("HD Capture (Preview Scaled 50%)", preview)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

    if key == 32:     # Capture on SPACE (Saves the FULL HD version)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        rgb_path = os.path.join(RGB_DIR, f"{ts}.png")
        depth_raw_path = os.path.join(DEPTH_RAW_DIR, f"{ts}.png")
        depth_cm_path = os.path.join(DEPTH_CM_DIR, f"{ts}.png")
        meta_path = os.path.join(META_DIR, f"{ts}.json")

        # Saving the original HIGH RES arrays (1280x720)
        cv2.imwrite(rgb_path, rgb)
        cv2.imwrite(depth_raw_path, depth_raw)
        cv2.imwrite(depth_cm_path, depth_colormap)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        meta = {
            "timestamp": ts,
            "resolution": "1280x720",
            "depth_scale_meters": depth_scale,
            "note": "Depth aligned to HD Color. Preview was scaled down, saved files are full HD."
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        print(f"Captured HD {ts}")

pipeline.stop()
cv2.destroyAllWindows()