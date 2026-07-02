import time
import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("runs/obb/train3/weights/best.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
pipeline.start(config)

print("Warming up...")
for _ in range(10):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    frame = np.asanyarray(color_frame.get_data())
    model(frame, imgsz=640)

inference_times = []
scene_stats = defaultdict(list)  
preprocess_times = []
model_inference_times = []
postprocess_times = []

def categorize_scene(num_objects):
    """Categorize based on number of detections."""
    if num_objects == 0:
        return "No detections"
    elif 1 <= num_objects <= 5:
        return "1–5 books"
    elif 6 <= num_objects <= 15:
        return "6–15 books"
    else:
        return "16+ books"

print("Starting inference loop... Press ESC to quit.")
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        start = time.perf_counter()
        results = model(frame, imgsz=640)
        end = time.perf_counter()

        speed = results[0].speed

        preprocess_times.append(speed['preprocess'])
        model_inference_times.append(speed['inference'])
        postprocess_times.append(speed['postprocess'])

        inference_time = end - start 
        fps = 1 / inference_time

        # Collect statistics
        if results and results[0].obb is not None:
            num_detections = len(results[0].obb)
        else:
            num_detections = 0
        category = categorize_scene(num_detections)

        inference_times.append(inference_time)
        scene_stats[category].append(inference_time)

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLOv8 OBB - RealSense", frame)

        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# Compute summary table
print("\n=== Inference Time Summary ===")
if inference_times:
    times_ms = np.array(inference_times) * 1000

    avg_time = np.mean(times_ms)
    std_time = np.std(times_ms)
    median_time = np.median(times_ms)
    min_time = np.min(times_ms)
    max_time = np.max(times_ms)

    print(f"Frames evaluated: {len(times_ms)}")
    print(f"Average inference time : {avg_time:.2f} ms")
    print(f"Median inference time  : {median_time:.2f} ms")
    print(f"Standard deviation     : {std_time:.2f} ms")
    print(f"Minimum inference time : {min_time:.2f} ms")
    print(f"Maximum inference time : {max_time:.2f} ms")
    print(f"Average FPS            : {1000/avg_time:.2f}")

    print("\n=== Ultralytics Internal Timing ===")
    print(f"Average preprocess : {np.mean(preprocess_times):.2f} ms")
    print(f"Average inference  : {np.mean(model_inference_times):.2f} ms")
    print(f"Average postprocess: {np.mean(postprocess_times):.2f} ms")

    order = ["No detections", "1–5 books", "6–15 books", "16+ books"]

    print("\nPer scene category:")
    for scene in order:
        if scene in scene_stats:
            avg_ms = np.mean(scene_stats[scene]) * 1000
            fps_scene = 1000 / avg_ms
            print(f"{scene}: {avg_ms:.1f} ms (FPS ≈ {fps_scene:.1f})")

    benchmark = pd.DataFrame({
        "inference_time_ms": np.array(inference_times) * 1000,
        "preprocess_ms": preprocess_times,
        "model_inference_ms": model_inference_times,
        "postprocess_ms": postprocess_times
    })

    benchmark.to_csv("yolo_inference_benchmark.csv", index=False)

    print("\nBenchmark saved as yolo_inference_benchmark.csv")
