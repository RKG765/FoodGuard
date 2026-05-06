"""
Redesigned clean diagrams for REPORT.md
White background, simple layout, large readable text, no emojis
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path("e:/BML/Semester-VI/Prj-3/results/diagrams")
OUT.mkdir(parents=True, exist_ok=True)

# ── Clean light theme ──────────────────────────────────────────────────────
BG       = "#FFFFFF"
LIGHT    = "#F6F8FA"
BORDER   = "#D0D7DE"
TEXT_D   = "#1F2328"   # dark text
TEXT_M   = "#57606A"   # muted
BLUE     = "#0969DA"
GREEN    = "#1A7F37"
RED      = "#CF222E"
ORANGE   = "#BC4C00"
PURPLE   = "#8250DF"
TEAL     = "#0A7080"
LBLUE    = "#DDF4FF"   # light blue fill
LGREEN   = "#DAFBE1"
LRED     = "#FFEBE9"
LORANGE  = "#FFF8C5"
LPURPLE  = "#FBEFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color":  TEXT_D,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
})

def save(fig, name, tight=True):
    path = OUT / f"{name}.png"
    if tight:
        plt.tight_layout(pad=1.5)
    fig.savefig(path, dpi=160, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  [OK] {path.name}")

def rbox(ax, cx, cy, w, h, text, fc=LBLUE, ec=BLUE, tc=TEXT_D, fs=11, bold=False):
    """Rounded rectangle with centred text."""
    patch = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.06",
                           facecolor=fc, edgecolor=ec, linewidth=2, zorder=3)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4,
            fontweight="bold" if bold else "normal",
            multialignment="center")

def arr(ax, x1, y1, x2, y2, color=BLUE, lw=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=16), zorder=2)

def hline(ax, y, label, color=TEXT_M, ls="--"):
    ax.axhline(y, color=color, ls=ls, lw=1.2, alpha=0.6)
    ax.text(ax.get_xlim()[1]*0.99, y + 0.02, label,
            color=color, fontsize=8.5, ha="right", va="bottom")

# ══════════════════════════════════════════════════════════════════════════
# 1. Threat flow — "What is FoodGuard trying to detect?"
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0, 13); ax.set_ylim(0, 6)
ax.axis("off")
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)

ax.text(6.5, 5.7, "What FoodGuard Detects — The 4 Image Classes",
        ha="center", va="center", fontsize=14, fontweight="bold", color=TEXT_D)

# Row 1: AI-generated path
rbox(ax, 2.0, 4.2, 2.8, 0.9, "Text Prompt\n(e.g. pizza photo)", LIGHT, BORDER, TEXT_D, 10)
rbox(ax, 5.5, 4.2, 2.8, 0.9, "Diffusion Model\n(RealVisXL / SDXL)", LBLUE, BLUE, BLUE, 10, True)
rbox(ax, 9.5, 5.0, 3.0, 0.8, "CLASS 1\nPerfect AI",   LGREEN, GREEN, GREEN, 10, True)
rbox(ax, 9.5, 4.0, 3.0, 0.8, "CLASS 2\nCompressed AI", LORANGE, ORANGE, ORANGE, 10, True)

arr(ax, 3.4, 4.2, 4.1, 4.2)
arr(ax, 6.9, 4.4, 7.9, 4.9, GREEN)
arr(ax, 6.9, 4.0, 7.9, 4.0, ORANGE)

ax.text(4.1+0.35, 4.35, "generates", fontsize=8.5, color=TEXT_M, style="italic")
ax.text(7.9+0.1, 4.7, "clean output", fontsize=8.5, color=GREEN, style="italic")
ax.text(7.9+0.1, 3.75, "re-saved (JPEG)", fontsize=8.5, color=ORANGE, style="italic")

# Row 2: Real photo path
rbox(ax, 2.0, 2.2, 2.8, 0.9, "Real Camera Photo\n(genuine food)", LGREEN, GREEN, GREEN, 10, True)
rbox(ax, 5.5, 2.2, 2.8, 0.9, "AI Inpainting\n(tampers 2-4% pixels)", LPURPLE, PURPLE, PURPLE, 10)
rbox(ax, 9.5, 2.2, 3.0, 0.8, "CLASS 3\nEdited AI",   LPURPLE, PURPLE, PURPLE, 10, True)
rbox(ax, 9.5, 1.1, 3.0, 0.8, "CLASS 4\nReal (genuine)",  LGREEN, GREEN, GREEN, 10, True)

arr(ax, 3.4, 2.2, 4.1, 2.2, PURPLE)
arr(ax, 6.9, 2.2, 7.9, 2.2, PURPLE)
ax.text(7.9+0.1, 2.05, "fraud evidence", fontsize=8.5, color=PURPLE, style="italic")

# Real arrow going directly to CLASS 4
ax.annotate("", xy=(7.9, 1.1), xytext=(3.4, 2.0),
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5,
                            connectionstyle="arc3,rad=0.3"), zorder=2)
ax.text(5.5, 0.9, "no tampering", fontsize=8.5, color=GREEN, style="italic", ha="center")

# Divider
ax.plot([0.3, 12.7], [3.2, 3.2], color=BORDER, lw=1, ls="--")
ax.text(0.4, 3.3, "AI-generated path", fontsize=8, color=TEXT_M)
ax.text(0.4, 3.05, "Real-photo path", fontsize=8, color=TEXT_M)

save(fig, "01_threat_flow")

# ══════════════════════════════════════════════════════════════════════════
# 2. Dataset distribution bar chart (cleaner than pie)
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Dataset Composition — 36,173 Images",
             fontsize=14, fontweight="bold", color=TEXT_D, y=1.01)

# Left: bar chart
ax = axes[0]
ax.set_facecolor(BG)
classes = ["Real", "Perfect AI", "Edited AI", "Compressed AI"]
counts  = [12000, 11173, 8000, 5000]
colors  = [GREEN, BLUE, PURPLE, ORANGE]
bars = ax.barh(classes, counts, color=colors, height=0.55, edgecolor="white", linewidth=1.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
            f"{count:,}", va="center", fontsize=11, fontweight="bold", color=TEXT_D)
ax.set_xlabel("Number of Images", fontsize=11)
ax.set_title("Image Count per Class", fontsize=12, fontweight="bold")
ax.set_xlim(0, 14500)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", labelsize=11)
ax.tick_params(axis="x", labelsize=9)
ax.grid(axis="x", color=BORDER, lw=0.8)

# Right: split table
ax2 = axes[1]
ax2.set_facecolor(BG)
ax2.axis("off")
ax2.set_title("Train / Val / Test Split (70/15/15)", fontsize=12, fontweight="bold")

col_labels = ["Class", "Train (70%)", "Val (15%)", "Test (15%)"]
row_data = [
    ["Real",          "8,400", "1,800", "1,800"],
    ["Perfect AI",    "7,821", "1,675", "1,677"],
    ["Edited AI",     "5,600", "1,200", "1,200"],
    ["Compressed AI", "3,500",   "750",   "750"],
    ["TOTAL",        "25,321", "5,425", "5,427"],
]
table = ax2.table(cellText=row_data, colLabels=col_labels,
                  cellLoc="center", loc="center",
                  bbox=[0, 0.05, 1, 0.9])
table.auto_set_font_size(False)
table.set_fontsize(10)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor(BORDER)
    if r == 0:
        cell.set_facecolor(BLUE); cell.set_text_props(color="white", fontweight="bold")
    elif r == len(row_data):
        cell.set_facecolor(LIGHT); cell.set_text_props(fontweight="bold")
    else:
        cell.set_facecolor(BG if r % 2 == 0 else LIGHT)

save(fig, "02_dataset_pie")

# ══════════════════════════════════════════════════════════════════════════
# 3. ELA pipeline — simple 4-step flow
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 4))
ax.set_xlim(0, 13); ax.set_ylim(0, 4)
ax.axis("off"); ax.set_facecolor(BG); fig.patch.set_facecolor(BG)

ax.text(6.5, 3.7, "Error Level Analysis (ELA) — How It Detects Tampered Images",
        ha="center", fontsize=13, fontweight="bold", color=TEXT_D)

steps = [
    (1.4, "Step 1\nInput Image\n(unknown origin)", LIGHT, BORDER),
    (4.2, "Step 2\nRe-save at\nJPEG Quality 95", LBLUE, BLUE),
    (7.0, "Step 3\nSubtract Original\nfrom Re-saved", LBLUE, BLUE),
    (9.8, "Step 4\nELA Map\n(difference image)", LBLUE, BLUE),
]
for x, lbl, fc, ec in steps:
    rbox(ax, x, 2.2, 2.2, 1.1, lbl, fc, ec, TEXT_D, 10)

for i in range(len(steps) - 1):
    arr(ax, steps[i][0] + 1.1, 2.2, steps[i+1][0] - 1.1, 2.2)

# Outcomes
rbox(ax, 11.9, 3.0, 2.0, 0.7, "Uniform grey\n= Real photo", LGREEN, GREEN, GREEN, 9)
rbox(ax, 11.9, 1.4, 2.0, 0.7, "Bright patch\n= Tampered area", LRED, RED, RED, 9)
arr(ax, 10.9, 2.55, 11.0, 2.9, GREEN)
arr(ax, 10.9, 1.85, 11.0, 1.55, RED)

save(fig, "03_ela_pipeline")

# ══════════════════════════════════════════════════════════════════════════
# 4. Full pipeline — clean 5-stage horizontal
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 4.5)
ax.axis("off"); ax.set_facecolor(BG); fig.patch.set_facecolor(BG)

ax.text(7, 4.2, "FoodGuard — Training Pipeline Overview",
        ha="center", fontsize=14, fontweight="bold", color=TEXT_D)

stages = [
    (1.2,  "1. Data Sources\n\nFood-101, UECFOOD256\nRealVisXL, Flux.1\nKandinsky, SDXL", LIGHT, BORDER),
    (3.8,  "2. 4-Class Dataset\n\n36,173 images\n70 / 15 / 15 split",                   LBLUE, BLUE),
    (6.6,  "3. Augmentation\n\nJPEG Degradation\nBlur + Flip + Jitter",                  LORANGE, ORANGE),
    (9.4,  "4. EfficientNet-B3\n\n512x512 input\n~12M parameters\nDropout 0.3",          LBLUE, BLUE),
    (12.2, "5. Deployment\n\nEMA Checkpoint\nThreshold = 0.50\nFPR = 1.56%",             LGREEN, GREEN),
]
for x, lbl, fc, ec in stages:
    rbox(ax, x, 2.2, 2.2, 2.8, lbl, fc, ec, TEXT_D, 9.5)

for i in range(len(stages) - 1):
    arr(ax, stages[i][0] + 1.1, 2.2, stages[i+1][0] - 1.1, 2.2)

# Loss label under arrow 3→4
ax.text(8.0, 1.55, "Focal Loss (gamma=2)\nAdamW + AMP + Grad Clip",
        ha="center", fontsize=8.5, color=TEXT_M, style="italic")

save(fig, "04_pipeline")

# ══════════════════════════════════════════════════════════════════════════
# 5. Degradation augmentation — branching from one input
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 5.5)
ax.axis("off"); ax.set_facecolor(BG); fig.patch.set_facecolor(BG)

ax.text(6, 5.2, "Degradation Augmentation — Why It Matters for AI Detection",
        ha="center", fontsize=13, fontweight="bold", color=TEXT_D)

rbox(ax, 1.4, 2.75, 2.0, 0.9, "Training\nImage", LIGHT, BORDER, TEXT_D, 11, True)
arr(ax, 2.4, 2.75, 3.0, 2.75)

# Branches
branches = [
    (4.8, 4.5, "25% — JPEG\nQuality 40-95",   LORANGE, ORANGE),
    (4.8, 3.3, "15% — Gaussian\nBlur (slight)", LBLUE,   BLUE),
    (4.8, 2.2, "10% — Sharpen\n(Unsharp Mask)", LPURPLE, PURPLE),
    (4.8, 1.0, "50% — No change\n(keep original)", LGREEN, GREEN),
]
for x, y, lbl, fc, ec in branches:
    ax.plot([3.0, 3.8, 3.8, x-0.9], [2.75, 2.75, y, y],
            color=BORDER, lw=1.5, zorder=1)
    arr(ax, x-0.9, y, x-1.0+0.9, y, ec)
    rbox(ax, x, y, 2.5, 0.8, lbl, fc, ec, TEXT_D, 10)

rbox(ax, 9.8, 2.75, 2.2, 0.9, "Augmented\nImage → Model", LGREEN, GREEN, GREEN, 11, True)
for _, y, _, _, ec in branches:
    ax.plot([6.05, 6.8, 6.8, 9.8-1.1], [y, y, 2.75, 2.75],
            color=BORDER, lw=1.5, zorder=1)

arr(ax, 8.7, 2.75, 8.8, 2.75, GREEN)

ax.text(6, 0.35, "Goal: Model learns underlying noise patterns, not specific JPEG compression levels",
        ha="center", fontsize=9, color=TEXT_M, style="italic")

save(fig, "05_degradation")

# ══════════════════════════════════════════════════════════════════════════
# 6. EMA — simple 3-box flow with annotation
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12); ax.set_ylim(0, 4)
ax.axis("off"); ax.set_facecolor(BG); fig.patch.set_facecolor(BG)

ax.text(6, 3.7, "EMA (Exponential Moving Average) — Why We Use It",
        ha="center", fontsize=13, fontweight="bold", color=TEXT_D)

rbox(ax, 2.0, 2.0, 3.2, 1.4,
     "Raw Training Weights\n\nUpdate every batch\nNoisy — can spike\nat end of epoch",
     LRED, RED, TEXT_D, 10)

rbox(ax, 6.0, 2.0, 3.2, 1.4,
     "EMA Shadow Weights\n\nSmooth running average\nFormula: 0.9998 x old\n+ 0.0002 x new",
     LORANGE, ORANGE, TEXT_D, 10)

rbox(ax, 10.0, 2.0, 3.2, 1.4,
     "Final Deployed Model\n\nStable, generalises well\nLower FPR on real\nworld images",
     LGREEN, GREEN, GREEN, 10)

arr(ax, 3.6, 2.0, 4.4, 2.0)
ax.text(4.0, 2.25, "updates after\nevery step", fontsize=8.5, color=TEXT_M,
        ha="center", style="italic")

arr(ax, 7.6, 2.0, 8.4, 2.0, GREEN)
ax.text(8.0, 2.25, "used for\nval + test", fontsize=8.5, color=GREEN,
        ha="center", style="italic")

ax.text(6.0, 0.55, "Result: The deployed checkpoint (food_ai_detector.pth) is immune to noisy final batches",
        ha="center", fontsize=9, color=TEXT_M, style="italic")

save(fig, "06_ema")

# ══════════════════════════════════════════════════════════════════════════
# 7. Windows challenges — clean problem/solution table
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.axis("off"); ax.set_facecolor(BG); fig.patch.set_facecolor(BG)

ax.text(0.5, 0.97, "Windows Training Environment — Problems & Solutions",
        ha="center", va="top", fontsize=13, fontweight="bold", color=TEXT_D,
        transform=ax.transAxes)

col_labels = ["Challenge", "Root Cause", "Solution", "Result"]
rows = [
    ["torch.compile() failed",
     "Triton (GPU kernel compiler)\nis Linux-only. Not available\non Windows natively.",
     "Disabled: USE_COMPILE = False\nNative PyTorch CUDA kernels\nused instead.",
     "No speed loss at this\ndataset scale. Training\nran stably."],
    ["num_workers=14 crashed\n(WinError 1455)",
     "Windows copies full CUDA DLLs\n(e.g. cublas64_12.dll) into the\nPaging File for each worker.",
     "Reverted to num_workers = 4\nPaging file can safely hold\n4 worker processes.",
     "GPU stays fully utilised.\nNo more paging file\nexhaustion errors."],
]

table = ax.table(
    cellText=rows, colLabels=col_labels,
    cellLoc="left", loc="center",
    bbox=[0.01, 0.05, 0.98, 0.82]
)
table.auto_set_font_size(False)
table.set_fontsize(9.5)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor(BORDER)
    cell.PAD = 0.08
    if r == 0:
        cell.set_facecolor(TEXT_D)
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
    elif r == 1:
        cell.set_facecolor(LRED if c < 2 else LGREEN)
    else:
        cell.set_facecolor(LRED if c < 2 else LGREEN)

save(fig, "07_windows")

# ══════════════════════════════════════════════════════════════════════════
# 8. Deployment — clean linear flow
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 4))
ax.set_xlim(0, 13); ax.set_ylim(0, 4)
ax.axis("off"); ax.set_facecolor(BG); fig.patch.set_facecolor(BG)

ax.text(6.5, 3.7, "Real-World Deployment Flow — FastAPI Inference",
        ha="center", fontsize=13, fontweight="bold", color=TEXT_D)

flow = [
    (1.3,  "Input\n\nFood photo from\ndelivery app\nor complaint form", LIGHT,   BORDER),
    (4.0,  "Preprocess\n\nResize to 512x512\nNormalize pixel\nvalues", LBLUE,   BLUE),
    (6.7,  "Inference\n\nEfficientNet-B3\n(EMA checkpoint)\n4-class softmax", LBLUE,   BLUE),
    (9.4,  "Filter\n\nP(real) > 0.50?\nYes = Accept\nNo = Flag", LORANGE, ORANGE),
    (12.1, "Output\n\nClass + Confidence\n+ ELA heatmap\n+ Action", LGREEN,  GREEN),
]
for x, lbl, fc, ec in flow:
    rbox(ax, x, 2.1, 2.2, 2.6, lbl, fc, ec, TEXT_D, 9.5)

for i in range(len(flow) - 1):
    arr(ax, flow[i][0] + 1.1, 2.1, flow[i+1][0] - 1.1, 2.1)

# Outcome labels
ax.text(9.4, 0.65, "Accept: genuine complaint\n     Flag: AI-generated fraud", 
        ha="center", fontsize=9, color=TEXT_M, style="italic")

save(fig, "08_deployment")

print("\nAll diagrams regenerated in:", OUT)
