import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("width_prediction_results.csv")

errors = df["error_mm"]

mean_err = errors.mean()
std_err = errors.std()

# Histogram of prediction errors
plt.figure()
plt.hist(errors, bins=20)
plt.axvline(mean_err)
plt.axvline(mean_err + 2*std_err)
plt.axvline(mean_err - 2*std_err)

plt.xlabel("Prediction Error (mm)")
plt.ylabel("Frequency")
plt.title("Width Prediction Error Distribution")

plt.show()

# Scatter plot of predicted vs true width
plt.figure()

plt.scatter(
    df["true_width_mm"],
    df["pred_width_mm"]
)

plt.plot(
    [df["true_width_mm"].min(), df["true_width_mm"].max()],
    [df["true_width_mm"].min(), df["true_width_mm"].max()]
)

plt.xlabel("True Width (mm)")
plt.ylabel("Predicted Width (mm)")
plt.title("Predicted vs True Book Width")

plt.show()
