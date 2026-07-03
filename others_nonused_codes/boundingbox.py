# fyp_capture_tool_with_bbox.py
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
from datetime import datetime

# ------------------------
# User config
# ------------------------
BASE_DIR = "D435_bbox"
RGB_DIR = os.path.join(BASE_DIR, "RGB")
DEPTH_RAW_DIR = os.path.join(BASE_DIR, "DEPTH_RAW")
DEPTH_CM_DIR = os.path.join(BASE_DIR, "DEPTH_COLORMAP")
META_DIR = os.path.join(BASE_DIR, "META")
INTR_DIR = os.path.join(BASE_DIR, "INTRINSICS")

MAX_DEPTH_M = 4.0         # software threshold (meters)
CENTER_PATCH_PAD = 0.20   # fraction padding for center patch inside bbox (for z_center median)
STRIP_FRACTION = 0.06     # fraction of bbox width for left/right strips for w calc
MIN_VALID_PIXELS = 10     # minimum valid depth pixels to compute median

# ------------------------
# Create folders
# ------------------------
for p in [RGB_DIR, DEPTH_RAW_DIR, DEPTH_CM_DIR, META_DIR, INTR_DIR]:
    os.makedirs(p, exist_ok=True)

# ------------------------
# Start RealSense
# ------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Depth sensor options
depth_sensor = profile.get_device().first_depth_sensor()
try:
    depth_sensor.set_option(rs.option.emitter_enabled, 1)
except Exception:
    pass
try:
    laser_range = depth_sensor.get_option_range(rs.option.laser_power)
    depth_sensor.set_option(rs.option.laser_power, laser_range.max)
except Exception:
    pass
try:
    depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
except Exception:
    pass

# Filters (no decimation to keep same resolution)
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
spatial.set_option(rs.option.holes_fill, 3)

temporal = rs.temporal_filter()
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal.set_option(rs.option.filter_smooth_delta, 20)

hole_filling = rs.hole_filling_filter()

# Align depth to color
align = rs.align(rs.stream.color)

# Save intrinsics once
color_stream = profile.get_stream(rs.stream.color)
color_intr = color_stream.as_video_stream_profile().get_intrinsics()
intrinsics = {
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
    json.dump(intrinsics, f, indent=4)
print("[INFO] intrinsics saved to", os.path.join(INTR_DIR, "camera_intrinsics.json"))

# ------------------------
# Interactive bbox drawing
# ------------------------
drawing = False
start_pt = None
current_bbox = None  # (x,y,w,h)

def mouse_cb(event, x, y, flags, param):
    global drawing, start_pt, current_bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_pt = (x, y)
        current_bbox = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        sx, sy = start_pt
        x0, y0 = min(sx, x), min(sy, y)
        x1, y1 = max(sx, x), max(sy, y)
        current_bbox = (x0, y0, x1 - x0, y1 - y0)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        sx, sy = start_pt
        x0, y0 = min(sx, x), min(sy, y)
        x1, y1 = max(sx, x), max(sy, y)
        current_bbox = (x0, y0, x1 - x0, y1 - y0)
        start_pt = None

cv2.namedWindow("RGB | Depth")
cv2.setMouseCallback("RGB | Depth", mouse_cb)

# Utility: deproject pixel (u,v) with depth Z_m to camera space
def deproject(intrinsics_obj, u, v, Z):
    # intrinsics_obj: profile.as_video_stream_profile().get_intrinsics()
    return rs.rs2_deproject_pixel_to_point(intrinsics_obj, [u, v], Z)

# Utility: compute median depth (meters) in a patch
def median_depth_in_patch(depth_arr, depth_scale, x1, y1, x2, y2):
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(depth_arr.shape[1], x2), min(depth_arr.shape[0], y2)
    patch = depth_arr[y1c:y2c, x1c:x2c].astype(np.uint16)
    if patch.size == 0:
        return np.nan
    valid = patch > 0
    if np.count_nonzero(valid) < MIN_VALID_PIXELS:
        return np.nan
    med = np.median(patch[valid])
    return float(med * depth_scale)

# Main loop
print("Instructions:")
print("- Draw bbox by left-click + drag on the RGB preview.")
print("- Press SPACE to capture current frame + current bbox measurements.")
print("- Press Q or ESC to quit.\n")

while True:
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Apply filters (no decimation)
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)

    # Convert to arrays
    depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
    color_image = np.asanyarray(color_frame.get_data())

    # Software threshold for max depth
    depth_scale = depth_sensor.get_depth_scale()
    depth_m = depth_raw.astype(np.float32) * depth_scale
    too_far = depth_m > MAX_DEPTH_M
    depth_display = depth_raw.copy()
    depth_display[too_far] = 0

    # Depth colormap for preview (use scaled 8-bit)
    depth_vis8 = cv2.convertScaleAbs(depth_display, alpha=0.03)  # tune alpha if needed
    depth_colormap = cv2.applyColorMap(depth_vis8, cv2.COLORMAP_JET)

    # Prepare overlay image for drawing
    overlay = color_image.copy()

    # If bbox exists, draw and compute measurements
    measured = {}
    if current_bbox is not None:
        x, y, w, h = current_bbox
        # Draw bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # center pixel
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        # central patch coords (avoid edges)
        pad = CENTER_PATCH_PAD
        cx1 = int(x + w * pad)
        cy1 = int(y + h * pad)
        cx2 = int(x + w * (1 - pad))
        cy2 = int(y + h * (1 - pad))

        z_center_m = median_depth_in_patch(depth_raw, depth_scale, cx1, cy1, cx2, cy2)
        if np.isfinite(z_center_m):
            # deproject center
            Xc, Yc, Zc = deproject(color_intr, cx, cy, z_center_m)
            measured['x_center_m'] = Xc
            measured['y_center_m'] = Yc
            measured['z_center_m'] = Zc
            # draw center cross
            cv2.drawMarker(overlay, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
            cv2.putText(overlay, f"Zc: {z_center_m:.3f} m", (x, y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(overlay, f"XYZ: {Xc:.3f},{Yc:.3f},{Zc:.3f} m", (x, y - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.putText(overlay, "Zc: N/A", (x, y - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # compute spine thickness w: deproject left & right mid-strips
        strip_w = max(2, int(w * STRIP_FRACTION))
        left_u1 = x
        left_u2 = x + strip_w
        right_u1 = x + w - strip_w
        right_u2 = x + w
        v_mid1 = int(y + h * 0.2)
        v_mid2 = int(y + h * 0.8)

        # left strip median depth
        zl = median_depth_in_patch(depth_raw, depth_scale, left_u1, v_mid1, left_u2, v_mid2)
        zr = median_depth_in_patch(depth_raw, depth_scale, right_u1, v_mid1, right_u2, v_mid2)
        if np.isfinite(zl) and np.isfinite(zr):
            # pick mid vertical pixel for deprojection
            v_mid = int((v_mid1 + v_mid2) / 2)
            ul = int((left_u1 + left_u2) / 2)
            ur = int((right_u1 + right_u2) / 2)
            Pl = np.array(deproject(color_intr, ul, v_mid, zl))
            Pr = np.array(deproject(color_intr, ur, v_mid, zr))
            w_m = float(np.linalg.norm(Pr - Pl))
            measured['w_m'] = w_m
            cv2.putText(overlay, f"w (thickness): {w_m*1000:.1f} mm", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "w: N/A", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # compute height h: top & bottom strips
        th = max(2, int(h * STRIP_FRACTION))
        top_v1 = y
        top_v2 = y + th
        bot_v1 = y + h - th
        bot_v2 = y + h
        u_mid1 = int(x + w * 0.2)
        u_mid2 = int(x + w * 0.8)

        zt = median_depth_in_patch(depth_raw, depth_scale, u_mid1, top_v1, u_mid2, top_v2)
        zb = median_depth_in_patch(depth_raw, depth_scale, u_mid1, bot_v1, u_mid2, bot_v2)
        if np.isfinite(zt) and np.isfinite(zb):
            u_mid = int((u_mid1 + u_mid2) / 2)
            vt = int((top_v1 + top_v2) / 2)
            vb = int((bot_v1 + bot_v2) / 2)
            Pt = np.array(deproject(color_intr, u_mid, vt, zt))
            Pb = np.array(deproject(color_intr, u_mid, vb, zb))
            h_m = float(np.linalg.norm(Pb - Pt))
            measured['h_m'] = h_m
            cv2.putText(overlay, f"h: {h_m*1000:.1f} mm", (x + w + 10, y + int(h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "h: N/A", (x + w + 10, y + int(h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Combine preview with depth colormap side-by-side (both 640x480)
    combined = np.hstack((overlay, depth_colormap))

    # Display
    cv2.imshow("RGB | Depth", combined)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        break

    if key == 32:  # SPACE -> save current frame + measured values + bbox
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rgb_path = os.path.join(RGB_DIR, f"{ts}.png")
        depth_raw_path = os.path.join(DEPTH_RAW_DIR, f"{ts}.png")
        depth_cm_path = os.path.join(DEPTH_CM_DIR, f"{ts}.png")
        meta_path = os.path.join(META_DIR, f"{ts}.json")

        # write images
        cv2.imwrite(rgb_path, color_image)
        # depth_raw is uint16; cv2.imwrite supports 16-bit PNG
        cv2.imwrite(depth_raw_path, depth_raw)
        cv2.imwrite(depth_cm_path, depth_colormap)

        # metadata
        meta = {
            "timestamp": ts,
            "intrinsics": intrinsics,
            "depth_scale_meters": depth_scale,
            "max_depth_m (software)": MAX_DEPTH_M,
            "bbox": None,
            "measurements": None
        }
        if current_bbox is not None:
            meta["bbox"] = {
                "x": int(current_bbox[0]),
                "y": int(current_bbox[1]),
                "w": int(current_bbox[2]),
                "h": int(current_bbox[3])
            }
            meta["measurements"] = measured

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        print(f"[SAVED] {ts}  bbox={meta['bbox']}  measurements={meta['measurements']}")

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
