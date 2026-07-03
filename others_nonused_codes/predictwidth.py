from ultralytics import YOLO
import joblib
import pandas as pd
import numpy as np

rf_D = joblib.load("width_regressor.pkl")
yolo = YOLO("C:\\Users\\Joseph\\Downloads\\yolov8_obb_project\\runs\\obb\\train3\\weights\\best.pt")

df = pd.read_csv("testlabels.csv")

family_map = {
    "pocket_small": 0,
    "A5_like": 1,
    "B5_like": 2,
    "large_textbook": 3
}

results_rows = []

for _, row in df.iterrows():
    img_path = row["filename"]
    true_width = row["W"]

    results = yolo(img_path)[0]

    if results.obb is None or len(results.obb) == 0:
        continue

    obb = results.obb[0]

    # --- Pixel height from OBB ---
    pts = obb.xyxyxyxy.reshape(4, 2)
    edges = [
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[1] - pts[2]),
        np.linalg.norm(pts[2] - pts[3]),
        np.linalg.norm(pts[3] - pts[0])
    ]
    h_px = max(edges)

    # --- CSV values ---
    height_true_mm = row["height_true_mm"]
    thickness_true_mm = row["thickness_true_mm"]
    paper_family_code = family_map[row["paper_family"]]
    D = row["D"]

    scale = height_true_mm / h_px
    T_norm = thickness_true_mm / height_true_mm

    X = pd.DataFrame(
        [[h_px, scale, T_norm, paper_family_code, D]],
        columns=[
            "height_px",
            "scale",
            "T_norm",
            "paper_family_code",
            "D"
        ]
    )

    pred_width = rf_D.predict(X)[0]
    error = pred_width - true_width

    results_rows.append({
        "filename": img_path,
        "true_width_mm": true_width,
        "pred_width_mm": pred_width,
        "error_mm": error,
        "abs_error_mm": abs(error)
    })

    print(
        f"{img_path} | "
        f"True: {true_width:.2f} | "
        f"Pred: {pred_width:.2f} | "
        f"Err: {error:.2f}"
    )

results_df = pd.DataFrame(results_rows)
results_df.to_csv("width_prediction_results.csv", index=False)

print("\nSaved: width_prediction_results.csv")
