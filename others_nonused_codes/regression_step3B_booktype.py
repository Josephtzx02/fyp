import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# =========================
# Load CSV
# =========================
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
    "book_type"
]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# =========================
# Encode book_type
# =========================
book_map = {
    "paperback": 0,
    "hardcover": 1,
    "notebook": 2
}
df["book_type_code"] = df["book_type"].map(book_map)

if df["book_type_code"].isna().any():
    raise ValueError("Unknown book_type detected")

# =========================
# Feature Engineering
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
        "book_type_code"
    ]
]

y = df["W"]

# =========================
# Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Random Forest
# =========================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("=========================")
print("Step 3B — RF + MANUAL book_type")
print(f"MAE (mm): {mae:.2f}")

# =========================
# Feature Importance
# =========================
fi = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:")
print(fi.sort_values(ascending=False))
