import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# =========================
# Load data
# =========================
df = pd.read_csv("labels2.csv")

# =========================
# Feature engineering
# =========================
df["scale"] = df["height_mm_est"] / df["height_px"]
df["T_norm"] = df["thickness_mm_est"] / df["height_mm_est"]

features = [
    "height_mm_est",
    "thickness_mm_est",
    "depth_rs_mm",
    "scale",
    "T_norm"
]

X = df[features]
y = df["W_mm"] 

# =========================
# Train / validation split
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Train Random Forest
# =========================
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

rf.fit(X_train, y_train)

# =========================
# Evaluate
# =========================
preds = rf.predict(X_val)
mae = mean_absolute_error(y_val, preds)

print("=========================")
print("Step 1 — Vision-only Geometry")
print(f"MAE (mm): {mae:.2f}")

# =========================
# Feature importance
# =========================
importances = pd.Series(
    rf.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importances)
