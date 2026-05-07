#!/usr/bin/env python3
"""Generate the small PR/ROC/CM/softmax icons used in the architecture diagram."""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


AIAA_RC = {
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":   "dejavuserif",
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "axes.linewidth":     0.7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2,
    "ytick.major.size":   2,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "savefig.dpi":        300,
    "figure.dpi":         300,
}

OUT = Path("paper_assets/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Class colors — match the safety-semantic palette used elsewhere
COL_NOM   = "#1A9850"   # green
COL_WARN  = "#E08214"   # orange
COL_HAZ   = "#D6604D"   # red
COL_DAN   = "#D6604D"   # red (binary alias)

# Curve colors (3 illustrative model qualities)
COL_GOOD = "#1B5E20"
COL_MID  = "#1565C0"
COL_LOW  = "#E65100"


def synthetic_roc(tpr_at_low_fpr, n=80):
    """Smooth ROC that hits TPR = tpr_at_low_fpr at FPR = 0.05."""
    fpr = np.linspace(0, 1, n)
    k = np.log((1 - 1e-3) / max(1 - tpr_at_low_fpr, 1e-3)) / 0.95
    tpr = 1 - (1 - tpr_at_low_fpr) * np.exp(-k * (fpr - 0.05))
    tpr = np.clip(tpr, fpr, 1.0)
    tpr[0], tpr[-1] = 0.0, 1.0
    return fpr, tpr


def synthetic_pr(prec_at_high_recall, n=80):
    """PR curve that hits precision = prec_at_high_recall at recall = 0.95."""
    rec = np.linspace(0, 1, n)
    base = 1.0
    decay = (1 - prec_at_high_recall) / 0.95 ** 2
    prec = base - decay * rec ** 2
    prec = np.clip(prec, 0.4, 1.0)
    prec[0] = 1.0
    return rec, prec


def _draw_roc(ax, with_legend=True, title=True):
    ax.plot([0, 1], [0, 1], color="#888", linestyle=":",
             linewidth=0.7, zorder=1)
    # Worst-first so each band fills only the new area the next-better
    # model gains on top of the previous one — non-overlapping pure colors.
    curves_in_order = [
        ("Model C", COL_LOW,  0.55),
        ("Model B", COL_MID,  0.80),
        ("Model A", COL_GOOD, 0.95),
    ]
    prev_tpr = None
    for name, color, tpr0 in curves_in_order:
        fpr, tpr = synthetic_roc(tpr0)
        lower = np.zeros_like(tpr) if prev_tpr is None else prev_tpr
        ax.fill_between(fpr, lower, tpr, color=color, alpha=0.30,
                         linewidth=0, zorder=2)
        ax.plot(fpr, tpr, color=color, linewidth=1.6, label=name, zorder=3)
        prev_tpr = tpr
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    ax.grid(color="#EEE", linewidth=0.4, linestyle="--", zorder=0)
    if title:
        ax.set_title("ROC curve", pad=4, loc="center")
    if with_legend:
        ax.legend(loc="lower right", frameon=False, fontsize=6.5,
                   handlelength=1.2, borderaxespad=0.2)


def _draw_pr(ax, with_legend=True, title=True):
    # Worst-first; each model's band fills only the additional area
    # gained over the previous model -- pure non-overlapping colors.
    curves_in_order = [
        ("Model C", COL_LOW,  0.60),
        ("Model B", COL_MID,  0.85),
        ("Model A", COL_GOOD, 0.95),
    ]
    prev_prec = None
    for name, color, p0 in curves_in_order:
        rec, prec = synthetic_pr(p0)
        lower = np.zeros_like(prec) if prev_prec is None else prev_prec
        ax.fill_between(rec, lower, prec, color=color, alpha=0.30,
                         linewidth=0, zorder=2)
        ax.plot(rec, prec, color=color, linewidth=1.6, label=name, zorder=3)
        prev_prec = prec
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0.45, 1.02)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.grid(color="#EEE", linewidth=0.4, linestyle="--", zorder=0)
    if title:
        ax.set_title("PR curve", pad=4, loc="center")
    if with_legend:
        ax.legend(loc="lower left", frameon=False, fontsize=6.5,
                   handlelength=1.2, borderaxespad=0.2)


def _draw_cm(ax, with_title=True):
    cm = np.array([
        [30, 2, 1],
        [3, 25, 6],
        [0, 4, 29],
    ], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    LABELS = ["Nominal", "Warning", "Hazard"]
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(3):
        for j in range(3):
            v = cm_norm[i, j]
            cnt = int(cm[i, j])
            tc = "white" if v > 0.55 else "#1a1a1a"
            ax.text(j, i, f"{cnt}\n({v:.0%})",
                    ha="center", va="center",
                    fontsize=6.5, color=tc,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS, fontsize=6.5)
    ax.set_yticklabels(LABELS, fontsize=6.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    if with_title:
        ax.set_title("Confusion matrix", pad=4, loc="center")
    ax.grid(False)


def _draw_softmax_binary(ax, with_threshold=True, with_title=True):
    """Bar chart: P(token | prompt) over a small slice of the vocabulary."""
    # 2 dominant + 8 background tokens
    tokens = [" the", " a", " ,", " is", " nominal", " on", " in", " danger", " to", " <eos>"]
    probs  = [0.005, 0.003, 0.002, 0.004, 0.27,    0.003, 0.002, 0.70,     0.003, 0.008]
    is_class = [t in (" nominal", " danger") for t in tokens]
    colors = []
    for tok, isc in zip(tokens, is_class):
        if tok == " nominal":
            colors.append(COL_NOM)
        elif tok == " danger":
            colors.append(COL_DAN)
        else:
            colors.append("#BBBBBB")

    x = np.arange(len(tokens))
    bars = ax.bar(x, probs, color=colors, edgecolor="white", linewidth=0.4,
                   width=0.7, zorder=3)

    # Annotate the two dominant bars with their probabilities
    for bar, tok, p in zip(bars, tokens, probs):
        if tok in (" nominal", " danger"):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{p:.2f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold",
                    color=bar.get_facecolor())

    if with_threshold:
        ax.axhline(0.5, color="#444", linestyle="--", linewidth=0.7, zorder=2)
        # Place the label on the left edge of the chart, above the line,
        # where there are no tall bars to collide with.
        ax.text(-0.4, 0.515, "decision threshold = 0.5",
                fontsize=6, color="#444", ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([t.strip() if t.strip() else t for t in tokens],
                        rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("P(token | prompt)")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color="#EEE", linewidth=0.4, linestyle="--", zorder=0)
    if with_title:
        ax.set_title("Softmax over vocabulary", pad=4, loc="center")


def _draw_softmax_multiclass(ax, with_title=True):
    """Same idea but with 3 dominant class tokens."""
    tokens = [" the", " nominal", " a", " warning", " is", " hazard", " in", " ,", " to", " <eos>"]
    probs  = [0.005, 0.18,       0.003, 0.32,       0.004, 0.45,      0.003, 0.002, 0.003, 0.030]
    color_map = {" nominal": COL_NOM, " warning": COL_WARN, " hazard": COL_HAZ}
    colors = [color_map.get(t, "#BBBBBB") for t in tokens]

    x = np.arange(len(tokens))
    bars = ax.bar(x, probs, color=colors, edgecolor="white", linewidth=0.4,
                   width=0.7, zorder=3)
    for bar, tok, p in zip(bars, tokens, probs):
        if tok in color_map:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{p:.2f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold",
                    color=bar.get_facecolor())

    ax.set_xticks(x)
    ax.set_xticklabels([t.strip() if t.strip() else t for t in tokens],
                        rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("P(token | prompt)")
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.grid(axis="y", color="#EEE", linewidth=0.4, linestyle="--", zorder=0)
    if with_title:
        ax.set_title("Softmax over vocabulary", pad=4, loc="center")


def make_roc_icon(path: Path):
    with mpl.rc_context(AIAA_RC):
        fig, ax = plt.subplots(figsize=(2.0, 2.0))
        _draw_roc(ax)
        ax.set_title("Binary --- ROC", pad=4, loc="center")
        fig.savefig(path); plt.close(fig)
    print(f"  saved -> {path}")


def make_pr_icon(path: Path):
    with mpl.rc_context(AIAA_RC):
        fig, ax = plt.subplots(figsize=(2.0, 2.0))
        _draw_pr(ax)
        ax.set_title("Binary --- PR", pad=4, loc="center")
        fig.savefig(path); plt.close(fig)
    print(f"  saved -> {path}")


def make_softmax_binary_icon(path: Path):
    with mpl.rc_context(AIAA_RC):
        fig, ax = plt.subplots(figsize=(3.0, 2.0))
        _draw_softmax_binary(ax)
        ax.set_title("Binary --- score distribution", pad=4, loc="center")
        fig.savefig(path); plt.close(fig)
    print(f"  saved -> {path}")


def make_cm_icon(path: Path):
    with mpl.rc_context(AIAA_RC):
        fig, ax = plt.subplots(figsize=(2.0, 2.0))
        _draw_cm(ax)
        ax.set_title("3-class --- confusion matrix", pad=4, loc="center")
        fig.savefig(path); plt.close(fig)
    print(f"  saved -> {path}")


def make_softmax_multiclass_icon(path: Path):
    with mpl.rc_context(AIAA_RC):
        fig, ax = plt.subplots(figsize=(3.0, 2.0))
        _draw_softmax_multiclass(ax)
        ax.set_title("3-class --- score distribution", pad=4, loc="center")
        fig.savefig(path); plt.close(fig)
    print(f"  saved -> {path}")


def make_binary_eval_combined(path: Path):
    """Three-panel figure: PR | ROC | Softmax. Together they tell the"""
    with mpl.rc_context(AIAA_RC):
        fig = plt.figure(figsize=(7.0, 2.2))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6], wspace=0.45)
        ax_pr  = fig.add_subplot(gs[0, 0])
        ax_roc = fig.add_subplot(gs[0, 1])
        ax_sm  = fig.add_subplot(gs[0, 2])
        _draw_pr (ax_pr)
        _draw_roc(ax_roc, with_legend=False)
        _draw_softmax_binary(ax_sm)
        ax_pr.set_title("(a) PR curve",  pad=4, loc="center")
        ax_roc.set_title("(b) ROC curve", pad=4, loc="center")
        ax_sm.set_title("(c) Score distribution", pad=4, loc="center")
        fig.suptitle("Binary classification (Nominal vs. Danger)",
                      fontsize=10, y=1.02)
        fig.savefig(path); plt.close(fig)
    print(f"  saved -> {path}")


def make_multiclass_eval_combined(path: Path):
    """Two-panel figure: Confusion matrix | Softmax."""
    with mpl.rc_context(AIAA_RC):
        fig = plt.figure(figsize=(5.5, 2.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.7], wspace=0.4)
        ax_cm = fig.add_subplot(gs[0, 0])
        ax_sm = fig.add_subplot(gs[0, 1])
        _draw_cm(ax_cm)
        _draw_softmax_multiclass(ax_sm)
        ax_cm.set_title("(a) Confusion matrix", pad=4, loc="center")
        ax_sm.set_title("(b) Score distribution", pad=4, loc="center")
        fig.suptitle("Three-class classification (Nominal / Warning / Hazard)",
                      fontsize=10, y=1.02)
        fig.savefig(path); plt.close(fig)
    print(f"  saved -> {path}")


def main():
    print("Generating task-symbol icons in paper_assets/figures/")
    # Standalone (drop-in glyphs, 2"x2" PDFs)
    make_roc_icon              (OUT / "icon_binary_roc.pdf")
    make_pr_icon               (OUT / "icon_binary_pr.pdf")
    make_softmax_binary_icon   (OUT / "icon_softmax_binary.pdf")
    make_cm_icon               (OUT / "icon_multiclass_cm.pdf")
    make_softmax_multiclass_icon(OUT / "icon_softmax_multiclass.pdf")
    # Combined (one figure per task)
    make_binary_eval_combined  (OUT / "icon_binary_eval.pdf")
    make_multiclass_eval_combined(OUT / "icon_multiclass_eval.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
