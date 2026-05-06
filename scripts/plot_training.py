"""
Training Dashboard — FoodGuard
================================
Plots a 4-panel figure from the actual training log:
  1. Loss curves  (train vs val)
  2. Accuracy curves  (train vs val + gap annotation)
  3. FPR on Real  (with ≤5% target line + threshold marker)
  4. Learning rate schedule

Output: results/training_dashboard.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Actual training log data ───────────────────────────────────────────────
epochs = list(range(1, 20))

train_loss = [1.4547, 0.5353, 0.3277, 0.2661, 0.2219, 0.2003, 0.1891,
              0.1767, 0.1716, 0.1693, 0.1626, 0.1568, 0.1522, 0.1479,
              0.1393, 0.1430, 0.1378, 0.1341, 0.1281]

val_loss   = [3.0968, 2.3801, 1.7207, 1.1987, 0.7852, 0.5218, 0.3669,
              0.3025, 0.2724, 0.2549, 0.2443, 0.2362, 0.2326, 0.2325,
              0.2391, 0.2474, 0.2597, 0.2707, 0.2792]

train_acc  = [64.80, 85.66, 92.72, 95.03, 96.30, 96.92, 97.35,
              97.68, 97.79, 97.91, 98.06, 98.20, 98.38, 98.52,
              98.86, 98.76, 98.83, 99.02, 99.16]

val_acc    = [37.29, 43.80, 51.94, 61.75, 73.88, 84.88, 91.80,
              94.69, 95.87, 96.06, 96.24, 96.13, 96.33, 96.07,
              95.89, 95.67, 95.32, 95.17, 95.19]

fpr        = [63.67, 56.56, 49.50, 38.67, 24.39, 11.28, 4.94,
              2.78,  1.89,  1.61,  1.39,  1.22,  1.06,  1.11,
              1.06,  1.06,  1.11,  1.11,  1.00]

lr         = [0.000120, 0.000210, 0.000300, 0.000299, 0.000296, 0.000291,
              0.000284, 0.000275, 0.000265, 0.000253, 0.000240, 0.000225,
              0.000209, 0.000193, 0.000176, 0.000159, 0.000141, 0.000124,
              0.000107]

gap = [t - v for t, v in zip(train_acc, val_acc)]

best_epoch    = 14   # epoch where val_loss was lowest (0.2325)
early_stop_ep = 19

# ── Style ──────────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_CLR  = "#30363d"
TRAIN_CLR = "#58a6ff"   # blue
VAL_CLR   = "#3fb950"   # green
GAP_CLR   = "#ffa657"   # orange
FPR_CLR   = "#f85149"   # red
LR_CLR    = "#d2a8ff"   # purple
TARGET_CLR= "#ff7b72"   # light red
BEST_CLR  = "#ffd700"   # gold

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_CLR,
    "axes.labelcolor":   "#c9d1d9",
    "axes.grid":         True,
    "grid.color":        GRID_CLR,
    "grid.linewidth":    0.6,
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "font.family":       "DejaVu Sans",
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  GRID_CLR,
    "legend.framealpha": 0.9,
})

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(
    "FoodGuard — EfficientNet-B3 Training Dashboard\n"
    "4-Class Food Fraud Detector  |  RTX 5070 Ti  |  AMP + Focal Loss + EMA",
    fontsize=14, fontweight="bold", color="#e6edf3", y=0.98
)
fig.patch.set_facecolor(DARK_BG)

def vline(ax, x, label, color=BEST_CLR, ls="--"):
    ax.axvline(x, color=color, linestyle=ls, linewidth=1.2, alpha=0.75)
    ax.text(x + 0.15, ax.get_ylim()[1] * 0.97, label,
            color=color, fontsize=8, va="top")

# ── Panel 1: Loss ──────────────────────────────────────────────────────────
ax1 = axes[0, 0]
ax1.plot(epochs, train_loss, color=TRAIN_CLR, lw=2,   marker="o", ms=4, label="Train Loss")
ax1.plot(epochs, val_loss,   color=VAL_CLR,   lw=2,   marker="s", ms=4, label="Val Loss")
ax1.axvline(best_epoch,  color=BEST_CLR, ls="--", lw=1.2, alpha=0.8)
ax1.axvline(early_stop_ep, color=FPR_CLR, ls=":",  lw=1.2, alpha=0.8)
ax1.text(best_epoch + 0.2,    max(val_loss) * 0.92, f"Best\n(ep {best_epoch})",
         color=BEST_CLR, fontsize=8, va="top")
ax1.text(early_stop_ep - 0.3, max(val_loss) * 0.75, "Early\nStop",
         color=FPR_CLR,  fontsize=8, va="top", ha="right")
ax1.set_title("Loss Curves", fontweight="bold", color="#e6edf3")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Focal Loss")
ax1.legend(loc="upper right")
ax1.set_xlim(0.5, 19.5)

# ── Panel 2: Accuracy + Gap ────────────────────────────────────────────────
ax2 = axes[0, 1]
ax2.plot(epochs, train_acc, color=TRAIN_CLR, lw=2, marker="o", ms=4, label="Train Acc")
ax2.plot(epochs, val_acc,   color=VAL_CLR,   lw=2, marker="s", ms=4, label="Val Acc")
ax2.fill_between(epochs, val_acc, train_acc, alpha=0.15, color=GAP_CLR, label="Train-Val Gap")

# Annotate final gap
ax2.annotate(
    f"Gap: {gap[-1]:.1f}%",
    xy=(epochs[-1], (train_acc[-1] + val_acc[-1]) / 2),
    xytext=(16, 80),
    arrowprops=dict(arrowstyle="->", color=GAP_CLR, lw=1.2),
    color=GAP_CLR, fontsize=9, fontweight="bold"
)
ax2.axhline(95, color="#8b949e", ls=":", lw=1, alpha=0.6)
ax2.text(1, 95.5, "95% baseline", color="#8b949e", fontsize=8)
ax2.axvline(best_epoch, color=BEST_CLR, ls="--", lw=1.2, alpha=0.8)
ax2.set_title("Accuracy Curves", fontweight="bold", color="#e6edf3")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(30, 101)
ax2.legend(loc="lower right")
ax2.set_xlim(0.5, 19.5)

# ── Panel 3: FPR on Real ──────────────────────────────────────────────────
ax3 = axes[1, 0]
ax3.plot(epochs, fpr, color=FPR_CLR, lw=2, marker="^", ms=5, label="FPR on Real (%)")
ax3.fill_between(epochs, fpr, 0, alpha=0.15, color=FPR_CLR)
ax3.axhline(5.0, color=TARGET_CLR, ls="--", lw=1.5,
            label="Target ≤ 5% FPR")

# Mark where FPR first drops under 5%
cross_epoch = next(i + 1 for i, f in enumerate(fpr) if f < 5.0)
ax3.axvline(cross_epoch, color=BEST_CLR, ls="--", lw=1.2, alpha=0.8)
ax3.text(cross_epoch + 0.2, 8,
         f"FPR < 5%\n(ep {cross_epoch})",
         color=BEST_CLR, fontsize=8)

# Final FPR annotation
ax3.annotate(
    f"Final FPR: {fpr[-1]:.2f}%",
    xy=(epochs[-1], fpr[-1]),
    xytext=(14, 15),
    arrowprops=dict(arrowstyle="->", color="#ffd700", lw=1.2),
    color="#ffd700", fontsize=9, fontweight="bold"
)
ax3.set_title("False Positive Rate on Real Images", fontweight="bold", color="#e6edf3")
ax3.set_xlabel("Epoch"); ax3.set_ylabel("FPR (%)")
ax3.set_ylim(-2, 70)
ax3.legend(loc="upper right")
ax3.set_xlim(0.5, 19.5)

# ── Panel 4: Learning Rate ────────────────────────────────────────────────
ax4 = axes[1, 1]
ax4.plot(epochs, [l * 1000 for l in lr], color=LR_CLR, lw=2,
         marker="D", ms=4, label="LR (×10⁻³)")
ax4.fill_between(epochs, [l * 1000 for l in lr], alpha=0.12, color=LR_CLR)

# Annotate warmup / cosine zones
ax4.axvspan(1, 3,  alpha=0.08, color="#58a6ff", label="Warmup (3 ep)")
ax4.axvspan(3, 19, alpha=0.05, color=LR_CLR,    label="Cosine Decay")
ax4.text(1.4,  0.28, "Warmup",  color="#58a6ff", fontsize=8, fontweight="bold")
ax4.text(9,    0.12, "Cosine Annealing", color=LR_CLR, fontsize=8)
ax4.axvline(best_epoch, color=BEST_CLR, ls="--", lw=1.2, alpha=0.8)

ax4.set_title("Learning Rate Schedule", fontweight="bold", color="#e6edf3")
ax4.set_xlabel("Epoch"); ax4.set_ylabel("LR (×10⁻³)")
ax4.legend(loc="upper right")
ax4.set_xlim(0.5, 19.5)

# ── Shared formatting ──────────────────────────────────────────────────────
for ax in axes.flat:
    ax.set_xticks(epochs)
    ax.tick_params(axis="both", labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# ── Save ───────────────────────────────────────────────────────────────────
out_dir = Path("e:/BML/Semester-VI/Prj-3/results")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "training_dashboard.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight",
            facecolor=DARK_BG, edgecolor="none")
plt.close()

print(f"[OK] Saved: {out_path}")
print(f"     Size: {out_path.stat().st_size // 1024} KB")
print()
print("Key milestones encoded in chart:")
print(f"  Epoch {cross_epoch:2d} — FPR first drops below 5%")
print(f"  Epoch {best_epoch:2d} — Best val loss (0.2325), model saved")
print(f"  Epoch 19   — Early stopping triggered (patience=5)")
print(f"  Final FPR  — {fpr[-1]:.2f}%  |  Test Acc — 96.26%")
