# Stage 5: Figure 1. Discount (DEV / HUM) per tokenizer, oldest vocabulary first, with 95% CI whiskers.
# Reads results/discount.csv; writes results/figure1_discount.pdf and .png (300 dpi).
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

os.chdir(next(d for d in ("/kaggle/working", "/content", ".") if os.path.isdir(d)))

# Colour plus marker shape per generation, so the figure still reads in greyscale print.
# The three colours were checked for colour-blind separation.
GENERATIONS = {
    "G1": ("#2a78d6", "o", "G1: before 2023"),
    "G2": ("#eb6834", "s", "G2: 2023-2024"),
    "G3": ("#1baf7a", "^", "G3: 2025-2026"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

d = pd.read_csv("results/discount.csv").sort_values(["year", "name"]).reset_index(drop=True)
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.4, 0.27 * len(d) + 1.2), layout="constrained")

ax.axvline(1.0, color=MUTED, linewidth=1, zorder=1, label="1.0: same cost")
for gen, (colour, marker, label) in GENERATIONS.items():
    g = d[d["gen"] == gen]
    if g.empty:
        continue
    whiskers = [g["discount_hum"] - g["discount_hum_low"], g["discount_hum_high"] - g["discount_hum"]]
    ax.errorbar(g["discount_hum"], g.index, xerr=whiskers, fmt=marker, color=colour, markersize=6.5,
                markeredgecolor="white", markeredgewidth=0.8, elinewidth=1.4, capsize=2.5, zorder=3, label=label)

ax.set_yticks(range(len(d)))
ax.set_yticklabels([f"{label} ({year})" for label, year in zip(d["label"], d["year"])], color=INK)
ax.set_ylim(len(d) - 0.5, -0.5)  # oldest at the top
# A ratio belongs on a log axis: halving and doubling the cost sit the same distance from 1.
ax.set_xscale("log")
low, high = d["discount_hum_low"].min() / 1.25, d["discount_hum_high"].max() * 1.25
ticks = [t for t in (0.125, 0.25, 0.5, 1, 2, 4, 8, 16) if low <= t <= high]
ax.set_xlim(min(low, 0.8), max(high, 1.25))
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks])
ax.minorticks_off()
side = ax.get_xaxis_transform()  # x in data units, y as a fraction of the axis height
ax.text(1.0, 1.01, "romanized costs more  ", transform=side, ha="right", va="bottom", color=MUTED, fontsize=7)
ax.text(1.0, 1.01, "  romanized saves tokens", transform=side, ha="left", va="bottom", color=MUTED, fontsize=7)
ax.set_xlabel("Discount (log scale)\nDEV tokens / HUM tokens", color=INK)
ax.tick_params(axis="x", colors=MUTED)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
fig.legend(loc="outside lower center", ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.2)

fig.savefig("results/figure1_discount.pdf")
fig.savefig("results/figure1_discount.png", dpi=300)
print("Saved results/figure1_discount.pdf and results/figure1_discount.png")
widest = (d["discount_hum_high"] - d["discount_hum_low"]).max()
print(f"Widest 95% interval: {widest:.3f}. If the whiskers hide behind the markers, say so in the caption.")
