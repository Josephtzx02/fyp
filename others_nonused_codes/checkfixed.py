import pandas as pd
import numpy as np

df = pd.read_csv("weight_dataset.csv")

# Estimate intrinsics from your dataset
df["fy_implied_depth"] = df["height_px"] * df["depth_rs_mm"] / df["height_mm_est"]
df["fx_implied_depth"] = df["thickness_px"] * df["depth_rs_mm"] / df["thickness_mm_est"]

fx = df["fx_implied_depth"].median()
fy = df["fy_implied_depth"].median()

# Reverse-calculate what depth was actually used
df["depth_used_from_H"] = df["height_mm_est"] * fy / df["height_px"]
df["depth_used_from_T"] = df["thickness_mm_est"] * fx / df["thickness_px"]

df["diff_to_rs"] = abs(df["depth_used_from_H"] - df["depth_rs_mm"])
df["diff_to_400"] = abs(df["depth_used_from_H"] - 400)

# likely hidden fixed rows
suspect_fixed = df[(df["diff_to_400"] < 0.5) & (df["diff_to_rs"] > 1.0)]

print("fx:", fx)
print("fy:", fy)
print(suspect_fixed[[
    "filename", "depth_rs_mm", "depth_used_from_H",
    "diff_to_rs", "diff_to_400",
    "H_mm", "height_mm_est"
]])