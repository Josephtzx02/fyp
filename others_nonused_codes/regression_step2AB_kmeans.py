import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

CSV_PATH = "labels.csv"
RANDOM_STATE = 42
N_CLUSTERS = 3

df = pd.read_csv(CSV_PATH) # Load dataset

df["width_mm"] = df["W"]

# FEATURE ENGINEERING
df["scale"] = df["height_mm_est"] / df["height_px"]
df["T_norm"] = df["thickness_mm_est"] / df["height_mm_est"]

features = [
    "height_true_mm",
    "thickness_true_mm",
    "depth_rs",
    "scale",
    "T_norm"
]

X = df[features]
y = df["width_mm"]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)

# =========================
# BASELINE RF
# =========================
rf_base = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE
)

rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)
mae_base = mean_absolute_error(y_test, y_pred_base)

print("=========================")
print("Step 2A - Baseline RF with true height and thickness")
print(f"MAE (mm): {mae_base:.2f}")

# =========================
# KMEANS CLUSTERING
# =========================
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10
)

df["book_cluster"] = kmeans.fit_predict(X)

Xc = df[features + ["book_cluster"]]
yc = df["width_mm"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.25, random_state=RANDOM_STATE
)

# =========================
# RF + CLUSTER
# =========================
rf_cluster = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE
)

rf_cluster.fit(Xc_train, yc_train)
yc_pred = rf_cluster.predict(Xc_test)
mae_cluster = mean_absolute_error(yc_test, yc_pred)

print("\n=========================")
print("Step 2B - Baseline RF + KMeans Cluster")
print(f"MAE (mm): {mae_cluster:.2f}")

# =========================
# COMPARISON
# =========================
print("\n=========================")
print("MAE Comparison")
print(f"Baseline MAE : {mae_base:.2f} mm")
print(f"Cluster MAE  : {mae_cluster:.2f} mm")
print(f"Improvement  : {mae_base - mae_cluster:.2f} mm")

# =========================
# FEATURE IMPORTANCE
# =========================
importances = pd.Series(
    rf_cluster.feature_importances_,
    index=features + ["book_cluster"]
).sort_values(ascending=False)

print("\nFeature Importance (Clustered Model):")
print(importances)
