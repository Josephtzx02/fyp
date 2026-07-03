import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# =========================
# 1. Load your CSV
# =========================
csv_path = "weight_dataset.csv" #width_dataset.csv
df = pd.read_csv(csv_path)

# Change these column names if your CSV uses different names
width_col = "W_mm"
weight_col = "mass"

# =========================
# 2. Function to plot bell curve
# =========================
def plot_distribution(data, title, xlabel):
    data = data.dropna()

    mean = data.mean()
    std = data.std()

    lower_1 = mean - std
    upper_1 = mean + std
    lower_2 = mean - 2 * std
    upper_2 = mean + 2 * std

    p_lower = np.percentile(data, 2.5)
    p_upper = np.percentile(data, 97.5)

    x = np.linspace(data.min(), data.max(), 300)
    y = norm.pdf(x, mean, std)

    plt.figure(figsize=(10, 6))

    # Histogram (PRIMARY: counts)
    counts, bins, _ = plt.hist(data, bins=12, alpha=0.6, edgecolor="black")

    # Scale PDF to match counts
    bin_width = bins[1] - bins[0]
    scaled_pdf = y * len(data) * bin_width

    # Plot scaled bell curve
    plt.plot(x, scaled_pdf, linewidth=2)

    # Mean and std lines
    plt.axvline(mean, linestyle="--", linewidth=2, label=f"Mean = {mean:.2f}")
    plt.axvline(lower_1, linestyle="--", label=f"-1σ = {lower_1:.2f}")
    plt.axvline(upper_1, linestyle="--", label=f"+1σ = {upper_1:.2f}")
    plt.axvline(p_lower, linestyle=":", label=f"2.5% = {p_lower:.2f}") #plt.axvline(lower_2, linestyle=":", label=f"-2σ = {lower_2:.2f}")
    plt.axvline(p_upper, linestyle=":", label=f"97.5% = {p_upper:.2f}") #plt.axvline(upper_2, linestyle=":", label=f"+2σ = {upper_2:.2f}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Books")  # PRIMARY axis

    # ===== Secondary axis (Density) =====
    ax = plt.gca()
    ax2 = ax.twinx()

    # Convert count scale back to density
    ax2.set_ylim(ax.get_ylim()[0] / (len(data) * bin_width),
                ax.get_ylim()[1] / (len(data) * bin_width))

    ax2.set_ylabel("Density")

    # Legend fix
    ax.legend(loc="upper right")

    plt.grid(True, alpha=0.3)
    plt.show()

    # Outlier check
    outliers = data[(data < lower_2) | (data > upper_2)]

    print("\n==============================")
    print(title)
    print("==============================")
    print(f"Sample size: {len(data)}")
    print(f"Mean: {mean:.2f}")
    print(f"Std Dev: {std:.2f}")
    print(f"Range: {data.min():.2f} to {data.max():.2f}")
    print(f"Suggested range (2.5%–97.5%): {p_lower:.2f} to {p_upper:.2f}") #print(f"Suggested normal range ±2σ: {lower_2:.2f} to {upper_2:.2f}")
    print(f"Number of possible outliers: {len(outliers)}")
    print(outliers)

# =========================
# 3. Plot width and weight
# =========================
plot_distribution(df[width_col], "Normal Distribution of Book Width", "Width (mm)")
plot_distribution(df[weight_col], "Normal Distribution of Book Weight", "Weight / Mass (g)")