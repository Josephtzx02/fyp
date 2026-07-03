import numpy as np
import pyrealsense2 as rs

# CONFIG
FIXED_Z_MM = 400.0
DEPTH_SCALE = 0.001  # meters → mm
MIN_VALID_PIXELS = 30
PATCH_RADIUS = 4
STRIP_HALF_WIDTH = 3

# HELPERS
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def median_depth_mm(depth_frame, cx, cy):
    if depth_frame is None:
        return None

    h, w = depth_frame.get_height(), depth_frame.get_width()
    depths = []

    for dy in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
        for dx in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
            x = clamp(cx + dx, 0, w - 1)
            y = clamp(cy + dy, 0, h - 1)
            d = depth_frame.get_distance(x, y)
            if d > 0:
                depths.append(d / DEPTH_SCALE)

    if len(depths) < MIN_VALID_PIXELS:
        return None

    return float(np.median(depths))

def strip_depth_mm(depth_frame, p1, p2):
    if depth_frame is None:
        return None

    length = int(np.linalg.norm(np.array(p2) - np.array(p1)))
    if length <= 0:
        return None

    depths = []

    for i in range(length):
        t = i / length
        x = int(p1[0] * (1 - t) + p2[0] * t)
        y = int(p1[1] * (1 - t) + p2[1] * t)

        for dx in range(-STRIP_HALF_WIDTH, STRIP_HALF_WIDTH + 1):
            d = depth_frame.get_distance(
                clamp(x + dx, 0, depth_frame.get_width() - 1),
                clamp(y, 0, depth_frame.get_height() - 1)
            )
            if d > 0:
                depths.append(d / DEPTH_SCALE)

    if len(depths) < MIN_VALID_PIXELS:
        return None

    return float(np.median(depths))

def deproject(px, z_mm, intrinsics):
    pt = rs.rs2_deproject_pixel_to_point(
        intrinsics,
        [float(px[0]), float(px[1])],
        z_mm * DEPTH_SCALE
    )
    return np.array(pt) * 1000.0

# MAIN API
def measure_obb(box_pts, depth_frame, intrinsics, use_depth):
    """
    Returns:
        valid, z_mm, width_mm, height_mm, mode
    """

    cx = int(np.mean(box_pts[:, 0]))
    cy = int(np.mean(box_pts[:, 1]))

    if not use_depth:
        z_mm = FIXED_Z_MM
        mode = "FIXED @400mm"
    else:
        z_mm = median_depth_mm(depth_frame, cx, cy)
        if z_mm is None:
            return None
        mode = "DEPTH-AWARE"

    # Midpoints
    top = ((box_pts[0] + box_pts[1]) / 2).astype(int)
    bottom = ((box_pts[2] + box_pts[3]) / 2).astype(int)
    left = ((box_pts[0] + box_pts[3]) / 2).astype(int)
    right = ((box_pts[1] + box_pts[2]) / 2).astype(int)

    z_h = strip_depth_mm(depth_frame, top, bottom) or z_mm
    z_w = strip_depth_mm(depth_frame, left, right) or z_mm

    p_top = deproject(top, z_h, intrinsics)
    p_bottom = deproject(bottom, z_h, intrinsics)
    p_left = deproject(left, z_w, intrinsics)
    p_right = deproject(right, z_w, intrinsics)

    height_mm = float(np.linalg.norm(p_top - p_bottom))
    width_mm = float(np.linalg.norm(p_left - p_right))

    return {
        "z_mm": z_mm,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "mode": mode
    }
