import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

CSV_PATH = "labels.csv"
df = pd.read_csv(CSV_PATH)

# =========================
# 2. Required columns check
# =========================
required = [
    "height_true_mm",
    "thickness_true_mm",
    "height_px",
    "W",
    "paper_family",
    "D" 
]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# =========================
# 3. Encode paper_family
# =========================
family_map = {
    "pocket_small": 0,
    "A5_like": 1,
    "B5_like": 2,
    "large_textbook": 3
}
df["paper_family_code"] = df["paper_family"].map(family_map)
if df["paper_family_code"].isna().any():
    raise ValueError("Unknown paper_family detected")

# =========================
# 4. Feature Engineering
# =========================
df["scale"] = df["height_true_mm"] / df["height_px"]
df["T_norm"] = df["thickness_true_mm"] / df["height_true_mm"]

# =========================
# 5. Step 5 Features
# =========================
X_pixel_plus_D = df[
    [
        "height_px",
        "scale",
        "T_norm",
        "paper_family_code",
        "D"
    ]
]

X_pixel_only = df[
    [
        "height_px",
        "scale",
        "D",
        "paper_family_code"
    ]
]

y = df["W"]

# =========================
# 6. Train/Test Split
# =========================
X_train_pix, X_test_pix, y_train, y_test = train_test_split(
    X_pixel_only, y, test_size=0.2, random_state=42
)

X_train_D, X_test_D, _, _ = train_test_split(
    X_pixel_plus_D, y, test_size=0.2, random_state=42
)

# =========================
# 7. Random Forest Model
# =========================
rf_pix = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
rf_pix.fit(X_train_pix, y_train)

rf_D = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
rf_D.fit(X_train_D, y_train)

# =========================
# 8. Evaluation
# =========================
y_pred_pix = rf_pix.predict(X_test_pix)
mae_pix = mean_absolute_error(y_test, y_pred_pix)

y_pred_D = rf_D.predict(X_test_D)
mae_D = mean_absolute_error(y_test, y_pred_D)

print("=========================")
print("Step 5B — Pixel + Measured Depth")
print(f"MAE (mm): {mae_D:.2f}")

print("\n=========================")
print("Step 5C — Pixel + Measured Depth - T-Norm")
print(f"MAE (mm): {mae_pix:.2f}")

# =========================
# 9. Feature Importance
# =========================
fi_pix = pd.Series(rf_pix.feature_importances_, index=X_pixel_only.columns)
fi_D = pd.Series(rf_D.feature_importances_, index=X_pixel_plus_D.columns)

print("\nFeature Importance — Pixel + D:")
print(fi_D.sort_values(ascending=False))

print("\nFeature Importance — Pixel + D - T-Norm:")
print(fi_pix.sort_values(ascending=False))