import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

models = [
    'Model A\n(Upper Bound)',
    'Model B\n(Estimated Dimensions)',
    'Model C\n(Raw Pixel Only)',
    'Model D\n(Optimized Pruned)'
]

ridge_mae = [97.86, 134.67, 142.36, 133.89]
huber_mae = [99.82, 137.03, 140.21, 131.65]
rf_mae = [107.03, 145.27, 148.86, 142.51]
extra_trees_mae = [106.14, 146.67, 150.32, 139.44]
grad_boost_mae = [115.57, 151.27, 150.48, 145.40]

x = np.arange(len(models))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(
    x - 2 * width,
    ridge_mae,
    width,
    label='Ridge Regression',
    color='#aec7e8',
    edgecolor='#6d7c91'
)

rects2 = ax.bar(
    x - width,
    huber_mae,
    width,
    label='Huber Regressor',
    color='#1f77b4',
    edgecolor='#134a70'
)

rects3 = ax.bar(
    x,
    rf_mae,
    width,
    label='Random Forest',
    color='#ff7f0e',
    edgecolor='#a65208'
)

rects4 = ax.bar(
    x + width,
    extra_trees_mae,
    width,
    label='Extra Trees Regressor',
    color='#2ca02c',
    edgecolor='#1b631b'
)

rects5 = ax.bar(
    x + 2 * width,
    grad_boost_mae,
    width,
    label='Gradient Boosting',
    color='#ffbb78',
    edgecolor='#a67a4e'
)

rects2[3].set_edgecolor('red')
rects2[3].set_linewidth(2.5)

ax.annotate(
    'Best deployment model\nHuber MAE = 131.65 g',
    xy=(x[3] - width, 133.5),
    xytext=(2.5, 156),
    arrowprops=dict(arrowstyle='->', linewidth=1.4),
    fontsize=10,
    fontweight='bold'
)

ax.set_ylabel(
    'Weight Mean Absolute Error (g)',
    fontsize=12,
    fontweight='bold'
)

ax.set_title(
    'Figure 4.4.3: Cross-Validation Weight Mean Absolute Error (MAE)\n'
    'Comparing Model A, Model B, Model C, and Model D Feature Layers',
    fontsize=13,
    fontweight='bold',
    pad=15
)

ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold')

ax.set_ylim(80, 160)

ax.grid(
    axis='y',
    linestyle=':',
    alpha=0.5,
    color='#bdc3c7'
)

for bars in [rects1, rects2, rects3, rects4, rects5]:
    ax.bar_label(
        bars,
        fmt='%.1f',
        padding=3,
        fontsize=8
    )

ax.legend(
    loc='upper left',
    frameon=True,
    facecolor='white',
    edgecolor='#bdc3c7',
    framealpha=0.95,
    fontsize=10
)

plt.tight_layout()

plt.savefig(
    'figure443.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("Figure 4.4.3 successfully updated.")