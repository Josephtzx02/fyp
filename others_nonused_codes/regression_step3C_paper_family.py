import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

CSV_PATH = "labels.csv"
df = pd.read_csv(CSV_PATH)

# =========================
# Required columns
# =========================
required = [
    "height_true_mm",
    "thickness_true_mm",
    "depth_rs",
    "height_px",
    "W",
    "paper_family"
]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# =========================
# Encode paper_family
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
# Feature Engineering (GROUND TRUTH SAFE)
# =========================
df["scale"] = df["height_true_mm"] / df["height_px"]
df["T_norm"] = df["thickness_true_mm"] / df["height_true_mm"]

X = df[
    [
        "height_true_mm",
        "thickness_true_mm",
        "depth_rs",
        "scale",
        "T_norm",
        "paper_family_code"
    ]
]

y = df["W"]  # true width

# =========================
# Train / Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Random Forest Model
# =========================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("=========================")
print("Step 3C — RF + PAPER FAMILY")
print(f"MAE (mm): {mae:.2f}")

# =========================
# Feature Importance
# =========================
fi = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:")
print(fi.sort_values(ascending=False))
