import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load thesis dataset
csv_path = 'weight_dataset_with_width_pred_mm.csv'
df = pd.read_csv(csv_path)

# Calculate residuals
df['residual'] = df['width_pred_mm'] - df['W_mm']

# Academic plotting style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

fig, ax = plt.subplots(figsize=(11, 6.5))

# Split Royal_UK from remaining paper families
other_mask = df['auto_paper_family'] != 'Royal_UK'
royal_mask = df['auto_paper_family'] == 'Royal_UK'

royal_df = df[royal_mask].copy()

true_royal = royal_df[royal_df['W_mm'] <= 169]
wide_textbook = royal_df[(royal_df['W_mm'] > 169) & (royal_df['W_mm'] < 200)]
square_outlier = royal_df[royal_df['W_mm'] >= 200]

# Scatter plots
ax.scatter(
    df.loc[other_mask, 'W_mm'],
    df.loc[other_mask, 'residual'],
    color='#bdc3c7',
    alpha=0.5,
    edgecolors='none',
    s=45,
    label='Other Paper Families'
)

ax.scatter(
    true_royal['W_mm'],
    true_royal['residual'],
    color='#2ecc71',
    alpha=0.9,
    edgecolors='#27ae60',
    s=60,
    marker='o',
    label='Royal-Like (145–169 mm)'
)

ax.scatter(
    wide_textbook['W_mm'],
    wide_textbook['residual'],
    color='#e74c3c',
    alpha=0.9,
    edgecolors='#c0392b',
    s=70,
    marker='s',
    label='Wide Textbook-Like (170–195 mm)'
)

ax.scatter(
    square_outlier['W_mm'],
    square_outlier['residual'],
    color='#9b59b6',
    alpha=0.9,
    edgecolors='#8e44ad',
    s=110,
    marker='X',
    label='Square Binder Outlier'
)

# Zero residual reference
ax.axhline(
    0,
    color='#2c3e50',
    linestyle='--',
    linewidth=2,
    alpha=0.85
)

# Linear trend line
coef = np.polyfit(df['W_mm'], df['residual'], 1)
trend = np.poly1d(coef)

x_fit = np.linspace(120, 240, 300)

ax.plot(
    x_fit,
    trend(x_fit),
    color='black',
    linestyle='--',
    linewidth=2,
    label='Linear Trend'
)

# Annotate extreme outlier
if not square_outlier.empty:
    outlier = square_outlier.loc[square_outlier['residual'].idxmin()]

    ax.annotate(
        '231 mm binder\nResidual = -71.1 mm',
        xy=(outlier['W_mm'], outlier['residual']),
        xytext=(210, -52),
        arrowprops=dict(
            arrowstyle='->',
            linewidth=1.2
        ),
        fontsize=10,
        fontweight='bold',
        ha='left'
    )

# Labels
ax.set_xlabel(
    'True Measured Book Width ($W_{mm}$) [mm]',
    fontsize=12,
    fontweight='bold'
)

ax.set_ylabel(
    'Width Prediction Residual (Predicted − Ground Truth) [mm]',
    fontsize=12,
    fontweight='bold'
)

# Uncomment if required
# ax.set_title(
#     'Figure 4.3: Distribution of width prediction out-of-fold residuals\n'
#     'showing tracking cluster constraints across the 205-book dataset',
#     fontsize=13,
#     fontweight='bold',
#     pad=15
# )

# Grid and axes
ax.grid(
    True,
    linestyle=':',
    alpha=0.6,
    color='#95a5a6'
)

ax.set_xlim(120, 240)
ax.set_ylim(-80, 40)

ax.minorticks_on()

# Legend
ax.legend(
    loc='upper right',
    frameon=True,
    facecolor='white',
    edgecolor='#bdc3c7',
    framealpha=0.95,
    fontsize=10
)

plt.tight_layout()

plt.savefig(
    'figure_4_3_residuals.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("Figure 4.3 successfully updated.")
