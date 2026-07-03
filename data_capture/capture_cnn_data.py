import cv2
import math
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import os
import pyrealsense2 as rs
import threading

# CONFIG
MODEL_PATH = "runs/obb/train3/weights/best.pt"
USE_REALSENSE = True
SOURCE = 2 # "book_obb_dataset_v3/images/train/20260117_162733_287050_jpg.rf.cc44722e0d74f044ec88e97d87b29588.jpg"
START_CONF = 0.8
CSV_OUTPUT = "width_weight_dataset.csv"
FIXED_DISTANCE_MM = 400.0
DISPLAY_SCALE = 0.75
INFO_PANEL_WIDTH = 420
CENTER_ZONE_RATIO = 0.15
CLASS_NAMES = {0: "book", 1: "shelf"}

# GLOBAL STATE
conf_threshold = START_CONF
detections = []
selected_det = None
USE_DEPTH_AWARE_MM = True
intrinsics = None
display_resize_scale = 1.0
WAITING_FOR_INPUT = False
INPUT_VALUES = None

os.makedirs("images", exist_ok=True) # FOLDERS

# UTILS: Four point transform for rotated crop
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    thicknessA = np.linalg.norm(br - bl)
    thicknessB = np.linalg.norm(tr - tl)
    maxThickness = int(max(thicknessA, thicknessB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0,0],[maxThickness-1,0],[maxThickness-1,maxHeight-1],[0,maxHeight-1]],dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxThickness, maxHeight))
    return warped

model = YOLO(MODEL_PATH) # Load model

# SAVE TO CSV
def save_to_csv(det, canvas):
    global WAITING_FOR_INPUT, INPUT_VALUES, PENDING_DET, PENDING_CANVAS

    if WAITING_FOR_INPUT:
        return

    WAITING_FOR_INPUT = True
    INPUT_VALUES = None
    PENDING_DET = det
    PENDING_CANVAS = canvas.copy()

    print("\n🟥 ENTER VALUES IN TERMINAL")
    threading.Thread(target=input_worker, daemon=True).start()


# CENTER CHECK
def check_center(cx, img_w):
    center_x = img_w / 2
    half_zone = (img_w * CENTER_ZONE_RATIO) / 2
    return abs(cx - center_x) <= half_zone

# POINT IN OBB
def point_in_obb(pt, obb_pts):
    return cv2.pointPolygonTest(obb_pts, pt, False) >= 0

# MOUSE CALLBACK
def mouse_callback(event, x, y, flags, param):
    global selected_det
    if WAITING_FOR_INPUT:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        rx = int(x / DISPLAY_SCALE / display_resize_scale)
        ry = int(y / DISPLAY_SCALE / display_resize_scale)
        for det in detections:
            if point_in_obb((rx, ry), det["box_pts"]):
                selected_det = det
                print(f"📌 Selected: {det['class_name']} | Conf={det['conf']:.3f}")
                break

cv2.namedWindow("OBB Demo", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("OBB Demo", mouse_callback)

def input_worker():
    global INPUT_VALUES, WAITING_FOR_INPUT
    try:
        D = float(input("Enter D (GT depth mm, camera→book): "))
        T = float(input("Enter T (thickness mm): "))
        H = float(input("Enter H (height mm): "))
        W = float(input("Enter W (width mm): "))
        M = float(input("Enter M (mass grams): "))
        INPUT_VALUES = (D, T, H, W, M)
    except:
        INPUT_VALUES = None
    WAITING_FOR_INPUT = False

# REALSENSE / VIDEO SETUP
cap = None
img = None
pipeline = None
align = None
depth_frame = None
IS_IMAGE = False

if USE_REALSENSE:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print("✅ RealSense initialized")
else:
    if isinstance(SOURCE, str) and SOURCE.lower().endswith((".jpg",".png",".jpeg")):
        IS_IMAGE = True
        img = cv2.imread(SOURCE)
        if img is None:
            raise RuntimeError(f"❌ Cannot load image: {SOURCE}")
        print(f"🖼 Loaded image: {SOURCE}")
    else:
        cap = cv2.VideoCapture(SOURCE)
        if not cap.isOpened():
            raise RuntimeError(f"❌ Cannot open video source: {SOURCE}")
        print(f"📹 Webcam opened: {SOURCE}")

# DEPTH UTILS
def get_median_depth_mm(depth_frame, cx, cy, patch=7):
    if depth_frame is None:
        return None
    half = patch // 2
    depths = []
    for dx in range(-half, half+1):
        for dy in range(-half, half+1):
            x = int(cx + dx)
            y = int(cy + dy)
            w = depth_frame.get_width()
            h = depth_frame.get_height()
            if x<0 or y<0 or x>=w or y>=h:
                continue
            try:
                d = depth_frame.get_distance(x,y)
                if d>0:
                    depths.append(d*1000)
            except: pass
    if len(depths)==0:
        return None
    return float(np.median(depths))

# MAIN LOOP
while True:
    # Frame capture
    
    if USE_REALSENSE:
        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            continue
        measure_canvas = np.asanyarray(color_frame.get_data())
    else:
        if IS_IMAGE:
            measure_canvas = img.copy()
        else:
            ret, measure_canvas = cap.read()
            if not ret:
                break

    detections.clear()

    if not WAITING_FOR_INPUT and INPUT_VALUES is not None:
        D, T, H, W, M = INPUT_VALUES

        spine_img = four_point_transform(PENDING_CANVAS, PENDING_DET["box_pts"])

        img_name = f"book_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join("book_width_dataset/images", img_name)
        cv2.imwrite(save_path, spine_img)

        row = {
        "filename": img_name,
        "depth_gt_mm": D,
        "depth_rs_mm": PENDING_DET["depth_mm"],
        "T_mm": T,
        "H_mm": H,
        "W_mm": W,
        "mass": M,
        "thickness_px": PENDING_DET["thickness_px"],
        "height_px": PENDING_DET["height_px"],
        "thickness_mm_est": PENDING_DET["thickness_mm"],
        "height_mm_est": PENDING_DET["height_mm"],
        "angle": angle_deg
        }
        df = pd.DataFrame([row])
        df.to_csv(CSV_OUTPUT, mode="a", header=not os.path.exists(CSV_OUTPUT), index=False)

        print(f"💾 Saved {img_name} with D={D}, T={T}, H={H}, W={W}")

        INPUT_VALUES = None

    # YOLO detection
    results = model(measure_canvas, conf=conf_threshold, verbose=False)[0]

    if results.obb:
        for box, conf, cls in zip(results.obb.xywhr, results.obb.conf, results.obb.cls):
            x,y,bt,bh,a = box.tolist()
            angle_deg = math.degrees(a)
            px_t = int(min(bt,bh))
            px_h = int(max(bt,bh))
            rect = ((x,y),(bt,bh),angle_deg)
            box_pts = cv2.boxPoints(rect).astype(int)
            is_center = check_center(x, measure_canvas.shape[1])
            zone = "CENTER" if is_center else "OFF-AXIS"
            depth_mm = get_median_depth_mm(depth_frame,x,y) if USE_REALSENSE else None
            depth_out_of_range = depth_mm is not None and depth_mm>400
            DEFAULT_FX, DEFAULT_FY = 600,600
            depth_ok = USE_DEPTH_AWARE_MM and depth_mm is not None and not depth_out_of_range and is_center and intrinsics
            if depth_ok:
                thickness_mm = px_t * depth_mm / intrinsics.fx
                height_mm = px_h * depth_mm / intrinsics.fy
                mode_used = "DEPTH-AWARE"
            else:
                thickness_mm = px_t * FIXED_DISTANCE_MM / (intrinsics.fx if intrinsics else DEFAULT_FX)
                height_mm = px_h * FIXED_DISTANCE_MM / (intrinsics.fy if intrinsics else DEFAULT_FY)
                mode_used = "FIXED @400mm"
            detections.append({
                "thickness_px": px_t,
                "height_px": px_h,
                "angle": angle_deg,
                "conf": float(conf),
                "class_name": CLASS_NAMES.get(int(cls), str(cls)),
                "box_pts": box_pts,
                "depth_mm": depth_mm,
                "depth_out_of_range": depth_out_of_range,
                "thickness_mm": thickness_mm,
                "height_mm": height_mm,
                "zone": zone,
                "mode_used": mode_used
            })

    # DRAWING
    display_canvas = measure_canvas.copy()
    for det in detections:
        color = (0,180,0) if det["zone"]=="CENTER" else (0,180,180)
        cv2.drawContours(display_canvas,[det["box_pts"]],0,color,2)
    if selected_det:
        sel_color = (0,255,0) if selected_det["zone"]=="CENTER" else (0,255,255)
        cv2.drawContours(display_canvas,[selected_det["box_pts"]],0,sel_color,3)

    h,w = display_canvas.shape[:2]
    display_resize_scale = min(1280/w, 720/h)
    display_canvas = cv2.resize(display_canvas,None,fx=display_resize_scale,fy=display_resize_scale,interpolation=cv2.INTER_AREA)
    display_img = cv2.resize(display_canvas,None,fx=DISPLAY_SCALE,fy=DISPLAY_SCALE,interpolation=cv2.INTER_AREA)
    if WAITING_FOR_INPUT:
        cv2.putText(
            display_img,
            "ENTER D, T, H, W, M IN TERMINAL",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            4
        )

    # INFO PANEL
    dh,dw = display_img.shape[:2]
    panel = np.zeros((dh,INFO_PANEL_WIDTH,3),dtype=np.uint8)
    panel[:]= (35,35,35)
    cv2.putText(panel,"Object Information",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)
    y=60; step=26
    if selected_det:
        lines=[
            f"Class        : {selected_det['class_name']}",
            f"Confidence   : {selected_det['conf']:.3f}",
            f"Thickness    : {selected_det['thickness_px']} px",
            f"Height       : {selected_det['height_px']} px",
            f"Angle        : {selected_det['angle']:.2f} deg",
            f"Depth (Z)    : {selected_det['depth_mm']:.1f} mm" if selected_det['depth_mm'] else "Depth : N/A",
            f"Zone         : {selected_det['zone']}",
            f"Thickness    : {selected_det['thickness_mm']:.1f} mm",
            f"Height       : {selected_det['height_mm']:.1f} mm",
            f"Mode         : {'DEPTH-AWARE' if USE_DEPTH_AWARE_MM else 'FIXED @400mm'}",
            f"RS Depth    : {selected_det['depth_mm']:.1f} mm" if selected_det['depth_mm'] else "RS Depth : N/A"
        ]
    else:
        lines=[
            "No object selected",
            "",
            "-> Click inside a book",
            "-> Press C to cancel",
            f"Mode: {'DEPTH-AWARE' if USE_DEPTH_AWARE_MM else 'FIXED @400mm'}"
        ]
    for line in lines:
        cv2.putText(panel,line,(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,(230,230,230),1)
        y+=step
    cv2.putText(panel,"W/X : Conf +/-   M : Mode",(20,dh-60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
    cv2.putText(panel,"C : Cancel   S : Save   Q : Quit",(20,dh-35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
    cv2.putText(panel,f"Current Conf: {conf_threshold:.2f}",(20,dh-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
    cv2.imshow("OBB Demo", np.hstack((display_img,panel)))

    # KEYS
    k = cv2.waitKey(1) & 0xFF
    if k==ord("q"):
        break
    elif k==ord("w"):
        conf_threshold = min(conf_threshold+0.05,0.95)
    elif k==ord("x"):
        conf_threshold = max(conf_threshold-0.05,0.05)
    elif k==ord("c"):
        selected_det=None
    elif k==ord("s") and selected_det and not WAITING_FOR_INPUT:
        save_to_csv(selected_det, measure_canvas)
#   elif k==ord("m"):
#       USE_DEPTH_AWARE_MM = not USE_DEPTH_AWARE_MM

# CLEAN EXIT
if USE_REALSENSE:
    pipeline.stop()
if cap:
    cap.release()
cv2.destroyAllWindows()
