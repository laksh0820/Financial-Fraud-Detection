import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata

trials = [
    {"alpha": 0.75, "gamma": 2.00, "val_f1": 0.3879},
    {"alpha": 0.45, "gamma": 3.75, "val_f1": 0.3375},
    {"alpha": 0.10, "gamma": 1.75, "val_f1": 0.2524},
    {"alpha": 0.20, "gamma": 0.75, "val_f1": 0.0000},
    {"alpha": 0.25, "gamma": 2.00, "val_f1": 0.3433},
    {"alpha": 0.40, "gamma": 3.00, "val_f1": 0.3635},
    {"alpha": 0.45, "gamma": 3.75, "val_f1": 0.3582},
    {"alpha": 0.25, "gamma": 4.50, "val_f1": 0.3598},
    {"alpha": 0.10, "gamma": 3.50, "val_f1": 0.2669},
    {"alpha": 0.45, "gamma": 3.00, "val_f1": 0.3659},
    {"alpha": 0.85, "gamma": 1.00, "val_f1": 0.4390},
    {"alpha": 0.85, "gamma": 0.50, "val_f1": 0.0000},
    {"alpha": 0.85, "gamma": 1.50, "val_f1": 0.4015},
    {"alpha": 0.65, "gamma": 1.25, "val_f1": 0.4443},  # best
    {"alpha": 0.70, "gamma": 1.00, "val_f1": 0.4344},
    {"alpha": 0.65, "gamma": 1.00, "val_f1": 0.4298},
    {"alpha": 0.60, "gamma": 2.50, "val_f1": 0.3993},
    {"alpha": 0.90, "gamma": 1.25, "val_f1": 0.4025},
    {"alpha": 0.60, "gamma": 2.25, "val_f1": 0.3839},
    {"alpha": 0.75, "gamma": 0.50, "val_f1": 0.0000},
]

alphas  = np.array([t["alpha"] for t in trials])
gammas  = np.array([t["gamma"] for t in trials])
val_f1s = np.array([t["val_f1"] for t in trials])

# ── Interpolate onto a regular grid for the contour ──
grid_alpha = np.linspace(0.10, 0.90, 200)
grid_gamma = np.linspace(0.50, 5.00, 200)
grid_a, grid_g = np.meshgrid(grid_alpha, grid_gamma)

grid_f1 = griddata(
    points=(alphas, gammas),
    values=val_f1s,
    xi=(grid_a, grid_g),
    method="cubic"   # smooth interpolation
)

# ── Plot ──
fig, ax = plt.subplots(figsize=(7, 5))

cf = ax.contourf(grid_a, grid_g, grid_f1, levels=15, cmap="YlOrRd")
cbar = fig.colorbar(cf, ax=ax)
cbar.set_label("Validation F1", fontsize=11)

# Scatter all trial points
sc = ax.scatter(alphas, gammas, c=val_f1s, cmap="YlOrRd",
                edgecolors="black", linewidths=0.6, s=60, zorder=5)

# Annotate trial numbers
for t in trials:
    ax.annotate(
        str(trials.index(t)),
        xy=(t["alpha"], t["gamma"]),
        xytext=(4, 4), textcoords="offset points",
        fontsize=7, color="black"
    )

# Mark the best trial with a star
best = max(trials, key=lambda t: t["val_f1"])
ax.scatter(best["alpha"], best["gamma"], marker="*", s=100,
           color="green", zorder=6, label=f'Best (α={best["alpha"]}, γ={best["gamma"]})')

ax.set_xlabel(r"$\alpha$", fontsize=13)
ax.set_ylabel(r"$\gamma$", fontsize=13)
ax.set_title("Focal Loss Hyperparameter Search\n(Bayesian Optimisation - 20 Trials, GIN, Small_HI)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig("bayes_opt_contour_gin_Small_HI.png", bbox_inches="tight", dpi=300)
plt.show()
print("Figure saved.")