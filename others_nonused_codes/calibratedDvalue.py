import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv("labels.csv")

X = data["depth_rs"].values.reshape(-1,1)
y = data["D"].values

reg = LinearRegression()
reg.fit(X, y)

k = reg.coef_[0]
b = reg.intercept_

print(f"Depth calibration: D_calibrated = {k:.4f} * depth_rs + {b:.2f}")