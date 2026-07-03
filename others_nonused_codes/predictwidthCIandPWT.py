import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

df = pd.read_csv("width_prediction_results.csv")

errors = df["error_mm"]
n = len(errors)

mean = errors.mean()
std = errors.std()
ci_95 = stats.t.interval(
    0.95,
    df=n-1,
    loc=mean,
    scale=std / np.sqrt(n)
)
tolerance = 3.0
pwt = (df["abs_error_mm"] <= tolerance).mean() * 100

print(f"PWT (±{tolerance} mm): {pwt:.1f}%") #Pass Within Tolerance
print(f"Mean Error: {mean:.2f} mm")
print(f"95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}] mm")