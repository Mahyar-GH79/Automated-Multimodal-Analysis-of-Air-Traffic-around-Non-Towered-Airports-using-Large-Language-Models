#!/usr/bin/env python3
"""Rebuild paper figures and tables from raw prediction JSONs."""

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D


PAGE_W   = 6.5     # AIAA conf usable width
HALF_W   = 3.2     # half-page figure
BODY_FS  = 10
LBL_FS   = 9
TICK_FS  = 8
LEG_FS   = 8
TITLE_FS = 10
DPI      = 300

AIAA_RC = {
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":   "dejavuserif",
    "font.size":          BODY_FS,
    "axes.titlesize":     TITLE_FS,
    "axes.labelsize":     LBL_FS,
    "xtick.labelsize":    TICK_FS,
    "ytick.labelsize":    TICK_FS,
    "legend.fontsize":    LEG_FS,
    "legend.title_fontsize": LEG_FS,
    "lines.linewidth":    1.2,
    "lines.markersize":   5,
    "patch.linewidth":    0.5,
    "axes.linewidth":     0.7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "axes.axisbelow":     True,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "figure.dpi":         DPI,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
}


MODEL_ORDER = ["qwen", "mistral", "gemma", "gpt-4o", "gpt-5.4",
               "claude-sonnet-4-6"]

OPEN_SOURCE = {"qwen", "mistral", "gemma"}
CLOSED_SOURCE = {"gpt-4o", "gpt-5.4", "claude-sonnet-4-6"}

MODEL_COLORS = {
    "qwen":              "#1F78B4",  # bright blue
    "mistral":           "#6A3D9A",  # bright purple
    "gemma":             "#33A02C",  # bright green
    "gpt-4o":            "#FF7F00",  # bright orange
    "gpt-5.4":           "#000000",  # black
    "claude-sonnet-4-6": "#E31A1C",  # bright red
}
MODEL_MARKERS = {
    "qwen": "o", "mistral": "s", "gemma": "^",
    "gpt-4o": "D", "gpt-5.4": "*", "claude-sonnet-4-6": "P",
}
MODEL_NAMES = {
    "qwen":              "Qwen 2.5-7B",
    "mistral":           "Mistral-7B",
    "gemma":             "Gemma-2-9B",
    "gpt-4o":            "GPT-4o",
    "gpt-5.4":           "GPT-5.4",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
}

STRATEGIES   = ["zero_shot", "one_shot", "few_shot"]
STRAT_NAMES  = {"zero_shot": "Zero-shot", "one_shot": "One-shot", "few_shot": "Few-shot"}
STRAT_SHORT  = {"zero_shot": "ZS",       "one_shot": "OS",       "few_shot": "FS"}
STRAT_COLORS = {"zero_shot": "#4393C3", "one_shot": "#F4A582", "few_shot": "#1B7837"}

# Safety-semantic palette for class colors (used in ASR per-class plot, etc.)
CLASS_COLORS = {
    "nominal": "#1A9850",   # green
    "warning": "#E08214",   # orange
    "hazard":  "#D6604D",   # red
    "danger":  "#D6604D",   # red (binary alias)
}

LABELS_3CLASS = ["nominal", "warning", "hazard"]
LABELS_BINARY = ["nominal", "danger"]

LABEL_NAMES = {"nominal": "Nominal", "warning": "Warning",
               "hazard": "Hazard", "danger": "Danger"}

HAZARD_LABEL_3 = {
    "simultaneous_final": "hazard", "wrong_runway_announcement": "hazard",
    "imc_vfr_conflict": "hazard", "runway_incursion_risk": "hazard",
    "missing_position_calls": "warning", "pattern_conflict": "warning",
    "silent_traffic": "warning", "go_around_conflict": "warning",
    "improper_entry": "warning",
    "nominal_single_aircraft": "nominal", "nominal_multi_aircraft": "nominal",
    "nominal_instrument_approach": "nominal",
}
HT_DISPLAY = {
    "simultaneous_final": "Simultaneous final",
    "wrong_runway_announcement": "Wrong runway",
    "imc_vfr_conflict": "IFR/VFR conflict",
    "runway_incursion_risk": "Runway incursion",
    "missing_position_calls": "Missing position calls",
    "pattern_conflict": "Pattern conflict",
    "silent_traffic": "Silent traffic (NORDO)",
    "go_around_conflict": "Go-around conflict",
    "improper_entry": "Improper entry",
    "nominal_single_aircraft": "Nominal -- single",
    "nominal_multi_aircraft": "Nominal -- multi",
    "nominal_instrument_approach": "Nominal -- instrument",
}


ROOT          = Path(".")
RAW_3CLASS    = ROOT / "results" / "raw"
RAW_BINARY    = ROOT / "results_binary" / "raw"
ABL_WHISPER   = ROOT / "results" / "ablation_whisper"
ABL_NOISE     = ROOT / "results" / "ablation_noise"
ABL_MASK      = ROOT / "results" / "ablation_mask"

OUT          = ROOT / "paper_assets"
OUT_FIG      = OUT / "figures"
OUT_TAB      = OUT / "tables"
OUT_FIG_M3   = OUT_FIG / "main_3class"
OUT_FIG_MB   = OUT_FIG / "main_binary"
OUT_FIG_AB   = OUT_FIG / "ablations"


def parse_run_key(rk: str):
    cot = rk.endswith("_cot")
    body = rk[:-4] if cot else rk
    for st in STRATEGIES:
        suf = "_" + st
        if body.endswith(suf):
            return body[:-len(suf)], st, cot
    raise ValueError(f"cannot parse {rk!r}")


def run_key(model, strategy, cot):
    return f"{model}_{strategy}{'_cot' if cot else ''}"


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path.relative_to(OUT)}")


def compute_run_metrics(records, classes):
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    n_correct = n_parse_ok = 0
    latencies = []
    for r in records:
        gt = r.get("ground_truth")
        pred = r.get("predicted")
        if gt in classes and pred in classes:
            i, j = classes.index(gt), classes.index(pred)
            cm[i, j] += 1
            if i == j:
                n_correct += 1
        if r.get("parse_ok"):
            n_parse_ok += 1
        if r.get("latency_s") is not None:
            latencies.append(r["latency_s"])
    n = len(records)
    accuracy = n_correct / n if n else 0.0

    per_class = {}
    f1s = precs = recs = 0
    f1_list, p_list, r_list = [], [], []
    for k, c in enumerate(classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
        per_class[c] = {"precision": p, "recall": r_, "f1": f1,
                         "support": int(cm[k, :].sum())}
        f1_list.append(f1); p_list.append(p); r_list.append(r_)

    return {
        "accuracy":  accuracy,
        "macro_p":   float(np.mean(p_list)),
        "macro_r":   float(np.mean(r_list)),
        "macro_f1":  float(np.mean(f1_list)),
        "parse_ok":  n_parse_ok / n if n else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n":         n,
        "avg_latency_s": float(np.mean(latencies)) if latencies else 0.0,
    }


def compute_per_hazard(records, classes):
    out = {}
    by_ht = defaultdict(list)
    for r in records:
        if (ht := r.get("hazard_type")):
            by_ht[ht].append(r)
    for ht, rs in by_ht.items():
        n = len(rs)
        n_correct = sum(1 for r in rs if r.get("ground_truth") == r.get("predicted"))
        out[ht] = {"accuracy": n_correct / n if n else 0.0, "n": n}
    return out


def rebuild_main_metrics(raw_dir: Path, classes):
    metrics, per_hazard = {}, {}
    for f in sorted(raw_dir.glob("*.json")):
        rk = f.stem
        try:
            data = json.load(open(f))
            records = data.get("records", [])
            if not records:
                continue
            model, strat, cot = parse_run_key(rk)
        except Exception:
            continue
        m = compute_run_metrics(records, classes)
        m.update({"model_key": model, "strategy": strat, "cot": cot})
        metrics[rk] = m
        per_hazard[rk] = compute_per_hazard(records, classes)
    return metrics, per_hazard


def _scores_from_records(records, positive_class):
    y_true, y_scores = [], []
    for r in records:
        gt = r.get("ground_truth")
        if gt is None:
            continue
        y_true.append(1 if gt == positive_class else 0)
        cs = r.get("class_scores") or {}
        if positive_class in cs:
            y_scores.append(float(cs[positive_class]))
        else:
            conf = float(r.get("confidence", 0.5))
            y_scores.append(conf if r.get("predicted") == positive_class
                              else 1.0 - conf)
    return y_true, y_scores


def _pr_curve(y_true, y_scores):
    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    P = sum(y_true) or 1
    tp = fp = 0
    precision, recall = [1.0], [0.0]
    last_score = None
    for s, y in pairs:
        if last_score is not None and s != last_score:
            precision.append(tp / max(tp + fp, 1))
            recall.append(tp / P)
        if y == 1:
            tp += 1
        else:
            fp += 1
        last_score = s
    precision.append(tp / max(tp + fp, 1))
    recall.append(tp / P)
    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i - 1]) * precision[i]
    return precision, recall, ap


def _roc_curve(y_true, y_scores):
    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    P = sum(y_true) or 1
    N = len(y_true) - sum(y_true) or 1
    tp = fp = 0
    fpr, tpr = [0.0], [0.0]
    last_score = None
    for s, y in pairs:
        if last_score is not None and s != last_score:
            fpr.append(fp / N)
            tpr.append(tp / P)
        if y == 1:
            tp += 1
        else:
            fp += 1
        last_score = s
    fpr.append(fp / N)
    tpr.append(tp / P)
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return fpr, tpr, auc


def fig_strategy_lines(metrics, classes, out_path: Path):
    """Two-panel line plot: Macro-F1 vs strategy. Direct + CoT."""
    with mpl.rc_context(AIAA_RC):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, HALF_W * 1.0),
                                  sharey=True)
        for pi, cot_flag in enumerate([False, True]):
            ax = axes[pi]
            x = np.arange(len(STRATEGIES))
            for m in MODEL_ORDER:
                ys = [metrics.get(run_key(m, s, cot_flag), {}).get("macro_f1")
                      for s in STRATEGIES]
                if not any(y is not None for y in ys):
                    continue
                yarr = [y if y is not None else np.nan for y in ys]
                ax.plot(x, yarr,
                         color=MODEL_COLORS[m],
                         marker=MODEL_MARKERS[m],
                         markersize=6, linewidth=1.4,
                         label=MODEL_NAMES[m])
            ax.set_xticks(x)
            ax.set_xticklabels([STRAT_NAMES[s] for s in STRATEGIES])
            ax.set_ylim(0.30, 1.00)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.4,
                     linestyle="--", zorder=0)
            ax.set_title(("(a) Direct prompting" if not cot_flag
                          else "(b) Chain-of-thought"),
                          pad=4, loc="center")
            if pi == 0:
                ax.set_ylabel(r"Macro-averaged F$_1$")
            if pi == 1:
                ax.legend(loc="lower right", frameon=True, framealpha=0.92,
                           edgecolor="#CCCCCC", fontsize=LEG_FS - 1)
        fig.tight_layout(pad=0.4)
        save_fig(fig, out_path)


def fig_perclass_heatmap(metrics, classes, out_path: Path):
    cols, col_labs, col_groups = [], [], []
    data = {l: [] for l in classes}
    for m in MODEL_ORDER:
        for cot_flag in [False, True]:
            for s in STRATEGIES:
                k = run_key(m, s, cot_flag)
                if k not in metrics:
                    continue
                cols.append(k)
                col_labs.append(STRAT_SHORT[s] + ("+CoT" if cot_flag else ""))
                col_groups.append(m)
                pc = metrics[k]["per_class"]
                for c in classes:
                    data[c].append(pc[c]["f1"])
    if not cols:
        return
    mat = np.array([data[l] for l in classes])
    K = len(classes)

    with mpl.rc_context(AIAA_RC):
        n_cols = len(cols)
        # Taller figure: more room below x-ticks for model-group labels.
        fig, ax = plt.subplots(figsize=(PAGE_W, 1.2 + K * 0.32))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
        cbar.ax.tick_params(labelsize=TICK_FS - 0.5)
        cbar.set_label(r"F$_1$", fontsize=LBL_FS)
        ax.set_yticks(range(K))
        ax.set_yticklabels([LABEL_NAMES[c] for c in classes])
        ax.set_xticks(range(n_cols))
        # Less aggressive rotation + slightly larger font for legibility.
        ax.set_xticklabels(col_labs, fontsize=TICK_FS - 1,
                            rotation=45, ha="right")
        ax.grid(False)

        # Vertical separators between models
        for i in range(1, n_cols):
            if col_groups[i] != col_groups[i - 1]:
                ax.axvline(i - 0.5, color="#888", linewidth=0.6, linestyle="--")

        # Model-group labels — placed BELOW the rotated tick labels so they
        # don't overlap with ZS/OS/FS/+CoT text.
        i = 0
        while i < n_cols:
            j = i
            while j < n_cols and col_groups[j] == col_groups[i]:
                j += 1
            cx = (i + j - 1) / 2
            ax.text(cx, -0.55, MODEL_NAMES[col_groups[i]],
                    ha="center", va="top",
                    fontsize=LBL_FS, fontweight="bold",
                    color=MODEL_COLORS[col_groups[i]],
                    transform=ax.get_xaxis_transform())
            i = j

        for i in range(K):
            for j in range(n_cols):
                v = mat[i, j]
                tc = "white" if (v < 0.25 or v > 0.78) else "#222"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color=tc)
        ax.set_title(r"Per-class F$_1$ across all runs", pad=8, loc="center")
        # Reserve space at the bottom for model-name row.
        fig.subplots_adjust(bottom=0.35)
        save_fig(fig, out_path)


def fig_per_hazard(per_hazard, metrics, classes, out_path: Path):
    """Best run per model on each hazard type."""
    best = {}
    for k, m in metrics.items():
        mk = m["model_key"]
        if mk not in best or m["macro_f1"] > best[mk][1]["macro_f1"]:
            best[mk] = (k, m)

    ht_order = sorted(HAZARD_LABEL_3.keys(),
                       key=lambda ht: ({"hazard": 0, "warning": 1, "nominal": 2}[
                                         HAZARD_LABEL_3[ht]], ht))
    is_binary = "danger" in classes

    with mpl.rc_context(AIAA_RC):
        fig, ax = plt.subplots(figsize=(PAGE_W, 6.5))
        models_present = [m for m in MODEL_ORDER if m in best]
        n_models = len(models_present)
        bar_h = 0.13
        offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_h * 1.05

        y_pos = []
        cur_y = 0.0
        prev_cls = None
        for ht in ht_order:
            cls = HAZARD_LABEL_3[ht]
            if prev_cls is not None and cls != prev_cls:
                cur_y += 0.5
            y_pos.append(cur_y)
            cur_y += 0.85
            prev_cls = cls
        y_pos = np.array(y_pos)

        for mi, m in enumerate(models_present):
            rk, _ = best[m]
            ph = per_hazard.get(rk, {})
            vals = [ph.get(ht, {}).get("accuracy", np.nan) for ht in ht_order]
            _, strat, cot = parse_run_key(rk)
            label = (f"{MODEL_NAMES[m]} ({STRAT_SHORT[strat]}"
                     + ("+CoT" if cot else "") + ")")
            ax.barh(y_pos + offsets[mi], vals,
                     height=bar_h * 0.95,
                     color=MODEL_COLORS[m],
                     edgecolor="white", linewidth=0.3,
                     label=label, zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([HT_DISPLAY[ht] for ht in ht_order],
                            fontsize=TICK_FS - 0.3)
        for tick, ht in zip(ax.get_yticklabels(), ht_order):
            cls = HAZARD_LABEL_3[ht]
            if is_binary and cls != "nominal":
                cls = "danger"
            tick.set_color(CLASS_COLORS[cls])
            tick.set_fontweight("bold")
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.set_xlabel("Accuracy")
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.5,
                 linestyle="--", zorder=0)
        # Dashed separator between every adjacent hazard-type row
        for i in range(len(y_pos) - 1):
            ax.axhline((y_pos[i] + y_pos[i + 1]) / 2.0,
                        color="#999999", linewidth=0.5,
                        linestyle=(0, (3, 3)), zorder=1, alpha=0.6)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92,
                   edgecolor="#CCCCCC", fontsize=LEG_FS - 1,
                   title="Best run per model")
        ax.set_title("Per-hazard-type accuracy", pad=5, loc="center")
        fig.tight_layout(pad=0.4)
        save_fig(fig, out_path)


def fig_latency_two_panel(metrics, out_path: Path):
    with mpl.rc_context(AIAA_RC):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, HALF_W * 1.0))

        ax0 = axes[0]
        models_present = [m for m in MODEL_ORDER
                           if any(v["model_key"] == m for v in metrics.values())]
        avg_lat = []
        for m in models_present:
            lats = [v["avg_latency_s"] for v in metrics.values()
                    if v["model_key"] == m and v["avg_latency_s"] > 0]
            avg_lat.append(float(np.mean(lats)) if lats else 0.0)
        names = [MODEL_NAMES[m] for m in models_present]
        cols  = [MODEL_COLORS[m] for m in models_present]
        bars = ax0.bar(names, avg_lat, width=0.6, color=cols,
                        edgecolor="white", linewidth=0.5, zorder=3)
        for b, lat in zip(bars, avg_lat):
            ax0.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                      f"{lat:.2f}s", ha="center", va="bottom",
                      fontsize=TICK_FS - 0.5, color="#222")
        ax0.set_ylabel("Avg. inference latency, s/scenario")
        ax0.set_ylim(0, max(avg_lat) * 1.25 if avg_lat else 1.0)
        ax0.tick_params(axis="x", labelrotation=20)
        ax0.grid(axis="y", color="#DDDDDD", linewidth=0.5,
                  linestyle="--", zorder=0)
        ax0.set_title("(a) Average latency", pad=4, loc="center")

        ax1 = axes[1]
        strat_marks = {"zero_shot": "o", "one_shot": "s", "few_shot": "^"}
        for m in MODEL_ORDER:
            for cot_flag in [False, True]:
                for s in STRATEGIES:
                    k = run_key(m, s, cot_flag)
                    v = metrics.get(k)
                    if not v or v["avg_latency_s"] <= 0:
                        continue
                    ax1.plot(v["avg_latency_s"], v["macro_f1"],
                              color=MODEL_COLORS[m],
                              marker=strat_marks[s],
                              fillstyle="full" if cot_flag else "none",
                              markersize=7, linestyle="none",
                              markeredgewidth=1.0, zorder=5)
        ax1.set_xlabel("Avg. latency, s/scenario")
        ax1.set_ylabel(r"Macro-averaged F$_1$")
        ax1.grid(color="#DDDDDD", linewidth=0.4, linestyle="--", zorder=0)

        h_models = [Line2D([0], [0], color=MODEL_COLORS[m], marker="o",
                             linestyle="none", markersize=5, label=MODEL_NAMES[m])
                     for m in models_present]
        h_strat = [Line2D([0], [0], color="#555", marker=strat_marks[s],
                            linestyle="none", markersize=5,
                            label=STRAT_NAMES[s])
                    for s in STRATEGIES]
        h_cot = [
            Line2D([0], [0], color="#555", marker="o", fillstyle="none",
                    linestyle="none", markersize=5, label="Direct"),
            Line2D([0], [0], color="#555", marker="o", fillstyle="full",
                    linestyle="none", markersize=5, label="CoT"),
        ]
        ax1.legend(handles=h_models + h_strat + h_cot,
                    loc="lower right", frameon=True, framealpha=0.92,
                    edgecolor="#CCCCCC", ncol=2, fontsize=6)
        ax1.set_title(r"(b) F$_1$ vs. latency", pad=4, loc="center")
        fig.tight_layout(pad=0.4)
        save_fig(fig, out_path)


def fig_cot_delta(metrics, out_path: Path):
    rows = []
    for m in MODEL_ORDER:
        for s in STRATEGIES:
            d = metrics.get(run_key(m, s, False), {}).get("macro_f1")
            c = metrics.get(run_key(m, s, True),  {}).get("macro_f1")
            if d is not None and c is not None:
                rows.append((m, s, c - d))
    if not rows:
        return

    with mpl.rc_context(AIAA_RC):
        fig, ax = plt.subplots(figsize=(PAGE_W, HALF_W * 0.95))
        x = np.arange(len(rows))
        deltas = [r[2] for r in rows]
        colors = [MODEL_COLORS[r[0]] for r in rows]
        bars = ax.bar(x, deltas, width=0.7, color=colors,
                       edgecolor="white", linewidth=0.5, zorder=3)
        for b, d in zip(bars, deltas):
            offs = 0.008 if d >= 0 else -0.018
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + offs,
                    f"{d:+.2f}", ha="center",
                    va="bottom" if d >= 0 else "top",
                    fontsize=TICK_FS - 1.5, color="#222")
        ax.axhline(0, color="#444", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([STRAT_SHORT[r[1]] for r in rows], fontsize=TICK_FS - 0.5)

        i = 0
        while i < len(rows):
            j = i
            while j < len(rows) and rows[j][0] == rows[i][0]:
                j += 1
            cx = (i + j - 1) / 2
            ax.text(cx, -0.18, MODEL_NAMES[rows[i][0]],
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=LBL_FS - 0.5,
                    color=MODEL_COLORS[rows[i][0]], fontweight="bold")
            if j < len(rows):
                ax.axvline(j - 0.5, color="#BBB", linewidth=0.6, linestyle="--")
            i = j

        ax.set_ylabel(r"$\Delta$ Macro-F$_1$ (CoT $-$ Direct)")
        deltas_arr = np.array(deltas)
        ymin, ymax = float(deltas_arr.min()) - 0.05, float(deltas_arr.max()) + 0.05
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.2f"))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.4,
                 linestyle="--", zorder=0)
        ax.set_title("Effect of chain-of-thought on F$_1$",
                      pad=18, loc="center")
        fig.tight_layout(pad=0.5)
        save_fig(fig, out_path)


def fig_pr_roc(raw_dir: Path, metrics, out_path: Path,
                positive_class="danger"):
    best = {}
    for rk, m in metrics.items():
        mk = m["model_key"]
        if mk not in best or m["macro_f1"] > best[mk][1]["macro_f1"]:
            best[mk] = (rk, m)

    curves = []
    for mk in MODEL_ORDER:
        if mk not in best:
            continue
        rk, m = best[mk]
        f = raw_dir / f"{rk}.json"
        if not f.exists():
            continue
        records = json.load(open(f)).get("records", [])
        y_true, y_scores = _scores_from_records(records, positive_class)
        if not y_true or sum(y_true) in (0, len(y_true)):
            continue
        prec, rec, ap   = _pr_curve(y_true, y_scores)
        fpr, tpr, auc   = _roc_curve(y_true, y_scores)
        sample = records[: min(5, len(records))]
        score_src = ("confidence" if any(
                         r.get("score_source") == "confidence_fallback"
                         for r in sample)
                     else "logprobs")
        curves.append((mk, prec, rec, ap, fpr, tpr, auc, score_src))

    if not curves:
        return

    with mpl.rc_context(AIAA_RC):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, HALF_W * 1.0))

        ax = axes[0]
        for mk, prec, rec, ap, fpr, tpr, auc, src in curves:
            label = f"{MODEL_NAMES[mk]} (AP={ap:.3f}"
            label += ", conf*)" if src == "confidence" else ")"
            ax.plot(rec, prec, color=MODEL_COLORS[mk],
                     linewidth=1.6, label=label)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.grid(color="#DDD", linewidth=0.4, linestyle="--", zorder=0)
        ax.set_title(f"(a) PR curve, positive = {positive_class}",
                      pad=4, loc="center")
        ax.legend(loc="lower left", frameon=True, framealpha=0.92,
                   edgecolor="#CCC", fontsize=LEG_FS - 1)

        ax = axes[1]
        ax.plot([0, 1], [0, 1], color="#888", linestyle=":", linewidth=0.8)
        for mk, prec, rec, ap, fpr, tpr, auc, src in curves:
            label = f"{MODEL_NAMES[mk]} (AUC={auc:.3f}"
            label += ", conf*)" if src == "confidence" else ")"
            ax.plot(fpr, tpr, color=MODEL_COLORS[mk],
                     linewidth=1.6, label=label)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.grid(color="#DDD", linewidth=0.4, linestyle="--", zorder=0)
        ax.set_title("(b) ROC curve", pad=4, loc="center")
        ax.legend(loc="lower right", frameon=True, framealpha=0.92,
                   edgecolor="#CCC", fontsize=LEG_FS - 1)

        if any(c[7] == "confidence" for c in curves):
            fig.text(0.5, -0.02,
                     "* Confidence-derived score (logprobs unavailable for "
                     "this model).",
                     ha="center", fontsize=LEG_FS - 1, style="italic",
                     color="#555")
        fig.tight_layout(pad=0.4)
        save_fig(fig, out_path)


def fig_confusion_matrix_grid(metrics, classes, out_path: Path):
    """2x3 grid of best run per model. Replaces the 36 per-run CM PDFs in the paper."""
    best = {}
    for rk, m in metrics.items():
        mk = m["model_key"]
        if mk not in best or m["macro_f1"] > best[mk][1]["macro_f1"]:
            best[mk] = (rk, m)

    K = len(classes)
    LABELS_CAP = [LABEL_NAMES[c] for c in classes]
    models_present = [m for m in MODEL_ORDER if m in best]
    n_rows, n_cols = (2, 3) if len(models_present) > 3 else (1, len(models_present))

    # Larger panels for legibility — full page width, height tuned to K so 2x2
    # binary grids and 3x3 multi-class grids both look proportional.
    panel_h = 2.3 if K == 2 else 2.7
    fig_h = panel_h * n_rows + 0.6   # extra room for suptitle/x labels
    cell_fs = TICK_FS + 1.5 if K == 2 else TICK_FS + 0.5

    with mpl.rc_context(AIAA_RC):
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(PAGE_W, fig_h),
                                  sharex=True, sharey=True)
        axes = np.array(axes).flatten()
        for ax_i, mk in enumerate(models_present):
            ax = axes[ax_i]
            rk, m = best[mk]
            cm = np.array(m["confusion_matrix"], dtype=float)
            rs = cm.sum(axis=1, keepdims=True).clip(min=1)
            cmn = cm / rs
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for i in range(K):
                for j in range(K):
                    v   = cmn[i, j]
                    cnt = int(cm[i, j])
                    ax.text(j, i, f"{cnt}\n({v:.0%})",
                             ha="center", va="center",
                             fontsize=cell_fs,
                             color="white" if v > 0.6 else "#1a1a1a",
                             fontweight="bold" if i == j else "normal")
            ax.set_xticks(range(K))
            ax.set_yticks(range(K))
            ax.set_xticklabels(LABELS_CAP, fontsize=TICK_FS)
            ax.set_yticklabels(LABELS_CAP, fontsize=TICK_FS)
            _, strat, cot = parse_run_key(rk)
            sub = (f"{MODEL_NAMES[mk]} -- {STRAT_NAMES[strat]}"
                   + (" + CoT" if cot else ""))
            ax.set_title(sub, fontsize=LBL_FS, pad=4, loc="center")
            ax.grid(False)
            if ax_i // n_cols == n_rows - 1:
                ax.set_xlabel("Predicted")
            if ax_i % n_cols == 0:
                ax.set_ylabel("True")

        for i in range(len(models_present), len(axes)):
            axes[i].axis("off")

        fig.suptitle("Confusion matrices (best run per model)",
                      fontsize=TITLE_FS, y=1.02)
        fig.tight_layout(pad=0.4)
        save_fig(fig, out_path)


ABL_MODEL_COLORS = {k: MODEL_COLORS[k] for k in OPEN_SOURCE}
ABL_MODEL_NAMES  = {k: MODEL_NAMES[k]  for k in OPEN_SOURCE}
ABL_MODEL_ORDER  = ["qwen", "mistral", "gemma"]


def _load_ablation(path: Path):
    return json.load(open(path))


def fig_whisper_compare(out_path: Path):
    """3-panel line plot: x = Whisper size, y = Macro-F1, line per model."""
    metrics = _load_ablation(ABL_WHISPER / "metrics" / "whisper_all_metrics.json")
    sizes = ["base", "medium", "large-v3"]            # NO GT
    cond_labs = [s.capitalize() if s != "large-v3" else "Large-v3"
                  for s in sizes]
    conditions = [f"whisper_{s}" for s in sizes]

    with mpl.rc_context(AIAA_RC):
        fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, HALF_W * 0.95),
                                  sharey=True)
        for pi, st in enumerate(STRATEGIES):
            ax = axes[pi]
            x = np.arange(len(sizes))
            for mk in ABL_MODEL_ORDER:
                ys = []
                for c in conditions:
                    key = f"{mk}_{st}_{c}"
                    ys.append(metrics.get(key, {}).get("macro_f1", np.nan))
                ax.plot(x, ys,
                         color=MODEL_COLORS[mk],
                         marker=MODEL_MARKERS[mk],
                         markersize=6, linewidth=1.3,
                         label=MODEL_NAMES[mk])
                for xi, yi in zip(x, ys):
                    if not np.isnan(yi):
                        ax.annotate(f"{yi:.2f}",
                                     xy=(xi, yi), xytext=(0, 6),
                                     textcoords="offset points",
                                     ha="center", fontsize=6,
                                     color=MODEL_COLORS[mk])
            ax.set_xticks(x)
            ax.set_xticklabels(cond_labs, fontsize=TICK_FS)
            ax.set_xlabel("Whisper model")
            ax.set_ylim(0.30, 0.85)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            ax.grid(axis="y", color="#DDD", linewidth=0.4,
                     linestyle="--", zorder=0)
            ax.set_title(f"({chr(97 + pi)}) {STRAT_NAMES[st]}",
                          pad=4, loc="center")
            if pi == 0:
                ax.set_ylabel(r"Macro-averaged F$_1$")
            if pi == 2:
                ax.legend(loc="lower right", frameon=True, framealpha=0.92,
                           edgecolor="#CCC", fontsize=LEG_FS - 1)
        fig.tight_layout(pad=0.4)
        save_fig(fig, out_path)


def fig_whisper_perclass(out_path: Path):
    """Per-class F1 across Whisper sizes (GT dropped, safety-semantic colors)."""
    metrics = _load_ablation(ABL_WHISPER / "metrics" / "whisper_all_metrics.json")
    sizes = ["base", "medium", "large-v3"]            # NO GT
    cond_labs = [s.capitalize() if s != "large-v3" else "Large-v3"
                  for s in sizes]
    conditions = [f"whisper_{s}" for s in sizes]
    classes = LABELS_3CLASS

    # For each model, average per-class F1 over strategies, then plot vs size.
    with mpl.rc_context(AIAA_RC):
        fig, axes = plt.subplots(1, len(ABL_MODEL_ORDER),
                                  figsize=(PAGE_W, HALF_W * 0.95),
                                  sharey=True)
        for pi, mk in enumerate(ABL_MODEL_ORDER):
            ax = axes[pi]
            x = np.arange(len(sizes))
            for c in classes:
                ys = []
                for cond in conditions:
                    f1s = []
                    for st in STRATEGIES:
                        key = f"{mk}_{st}_{cond}"
                        f = metrics.get(key, {}).get("per_class", {}) \
                                    .get(c, {}).get("f1")
                        if f is not None:
                            f1s.append(f)
                    ys.append(np.mean(f1s) if f1s else np.nan)
                ax.plot(x, ys,
                         color=CLASS_COLORS[c],
                         marker={"nominal":"o","warning":"s","hazard":"^"}[c],
                         markersize=6, linewidth=1.3,
                         label=LABEL_NAMES[c])
            ax.set_xticks(x)
            ax.set_xticklabels(cond_labs, fontsize=TICK_FS)
            ax.set_xlabel("Whisper model")
            ax.set_ylim(0.0, 1.05)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            ax.grid(axis="y", color="#DDD", linewidth=0.4,
                     linestyle="--", zorder=0)
            ax.set_title(f"({chr(97 + pi)}) {MODEL_NAMES[mk]}",
                          pad=4, loc="center")
            if pi == 0:
                ax.set_ylabel(r"Per-class F$_1$")
            if pi == len(ABL_MODEL_ORDER) - 1:
                ax.legend(loc="lower right", frameon=True, framealpha=0.92,
                           edgecolor="#CCC", fontsize=LEG_FS - 1,
                           title="Class")
        fig.tight_layout(pad=0.4)
        save_fig(fig, out_path)


def fig_noise_curve(out_path: Path):
    """Macro-F1 vs noise level (NSR%), one line per model."""
    metrics = _load_ablation(ABL_NOISE / "metrics" / "noise_all_metrics.json")
    nsrs = [5, 10, 25, 50, 75]
    conditions = [f"noise_{n}" for n in nsrs]

    with mpl.rc_context(AIAA_RC):
        fig, ax = plt.subplots(figsize=(HALF_W * 1.4, HALF_W * 1.0))
        for mk in ABL_MODEL_ORDER:
            ys = []
            for c in conditions:
                # noise ablation runs only zero-shot
                key = f"{mk}_zero_shot_{c}"
                ys.append(metrics.get(key, {}).get("macro_f1", np.nan))
            ax.plot(nsrs, ys,
                     color=MODEL_COLORS[mk],
                     marker=MODEL_MARKERS[mk],
                     markersize=6, linewidth=1.3,
                     label=MODEL_NAMES[mk])
        ax.set_xlabel("Noise-to-signal ratio, %")
        ax.set_ylabel(r"Macro-averaged F$_1$")
        ax.set_ylim(0.20, 0.85)
        ax.set_xticks(nsrs)
        ax.grid(axis="y", color="#DDD", linewidth=0.4,
                 linestyle="--", zorder=0)
        ax.legend(loc="lower left", frameon=True, framealpha=0.92,
                   edgecolor="#CCC", fontsize=LEG_FS - 1)
        ax.set_title("Audio noise robustness (zero-shot, Whisper-large-v3)",
                      pad=4, loc="center")
        fig.tight_layout(pad=0.4)
        save_fig(fig, out_path)


def fig_noise_perclass(out_path: Path):
    metrics = _load_ablation(ABL_NOISE / "metrics" / "noise_all_metrics.json")
    nsrs = [5, 10, 25, 50, 75]
    conditions = [f"noise_{n}" for n in nsrs]
    classes = LABELS_3CLASS

    with mpl.rc_context(AIAA_RC):
        fig, axes = plt.subplots(1, len(ABL_MODEL_ORDER),
                                  figsize=(PAGE_W, HALF_W * 0.9),
                                  sharey=True)
        for pi, mk in enumerate(ABL_MODEL_ORDER):
            ax = axes[pi]
            for c in classes:
                ys = []
                for cond in conditions:
                    key = f"{mk}_zero_shot_{cond}"
                    f = metrics.get(key, {}).get("per_class", {}) \
                                .get(c, {}).get("f1")
                    ys.append(f if f is not None else np.nan)
                ax.plot(nsrs, ys,
                         color=CLASS_COLORS[c],
                         marker={"nominal":"o","warning":"s","hazard":"^"}[c],
                         markersize=6, linewidth=1.3,
                         label=LABEL_NAMES[c])
            ax.set_xlabel("NSR, %")
            ax.set_xticks(nsrs)
            ax.set_ylim(0.0, 1.05)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            ax.grid(axis="y", color="#DDD", linewidth=0.4,
                     linestyle="--", zorder=0)
            ax.set_title(f"({chr(97 + pi)}) {MODEL_NAMES[mk]}",
                          pad=4, loc="center")
            if pi == 0:
                ax.set_ylabel(r"Per-class F$_1$")
            if pi == len(ABL_MODEL_ORDER) - 1:
                ax.legend(loc="lower left", frameon=True, framealpha=0.92,
                           edgecolor="#CCC", fontsize=LEG_FS - 1,
                           title="Class")
        fig.tight_layout(pad=0.4)
        save_fig(fig, out_path)


def fig_mask_curve(out_path: Path):
    """Macro-F1 vs masking rate, separate panels for word vs utterance."""
    metrics = _load_ablation(ABL_MASK / "metrics" / "mask_all_metrics.json")
    rates = [10, 20, 40, 60, 80]
    mtypes = ["word", "utterance"]

    with mpl.rc_context(AIAA_RC):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, HALF_W * 1.0),
                                  sharey=True)
        for pi, mt in enumerate(mtypes):
            ax = axes[pi]
            for mk in ABL_MODEL_ORDER:
                ys = []
                for r in rates:
                    key = f"{mk}_zero_shot_mask_{mt}_{r}"
                    ys.append(metrics.get(key, {}).get("macro_f1", np.nan))
                ax.plot(rates, ys,
                         color=MODEL_COLORS[mk],
                         marker=MODEL_MARKERS[mk],
                         markersize=6, linewidth=1.3,
                         label=MODEL_NAMES[mk])
            ax.set_xlabel("Masking rate, %")
            ax.set_xticks(rates)
            ax.set_ylim(0.0, 0.85)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            ax.grid(axis="y", color="#DDD", linewidth=0.4,
                     linestyle="--", zorder=0)
            ax.set_title(f"({chr(97 + pi)}) {mt.capitalize()} masking",
                          pad=4, loc="center")
            if pi == 0:
                ax.set_ylabel(r"Macro-averaged F$_1$")
            if pi == 1:
                ax.legend(loc="lower left", frameon=True, framealpha=0.92,
                           edgecolor="#CCC", fontsize=LEG_FS - 1)
        fig.tight_layout(pad=0.4)
        save_fig(fig, out_path)


def fig_mask_heatmap(out_path: Path):
    """Heatmap of Macro-F1 across (model, masking rate) for word & utterance."""
    metrics = _load_ablation(ABL_MASK / "metrics" / "mask_all_metrics.json")
    rates = [10, 20, 40, 60, 80]
    mtypes = ["word", "utterance"]
    n_rates = len(rates)

    # Build (n_models, n_rates) for each masking type
    mat_by_type = {}
    for mt in mtypes:
        mat = np.full((len(ABL_MODEL_ORDER), n_rates), np.nan)
        for i, mk in enumerate(ABL_MODEL_ORDER):
            for j, r in enumerate(rates):
                key = f"{mk}_zero_shot_mask_{mt}_{r}"
                v = metrics.get(key, {}).get("macro_f1")
                if v is not None:
                    mat[i, j] = v
        mat_by_type[mt] = mat

    with mpl.rc_context(AIAA_RC):
        # gridspec with extra column on the right for colorbar; controls overlap
        fig, axes = plt.subplots(
            1, 3,
            figsize=(PAGE_W, HALF_W * 0.85),
            gridspec_kw={"width_ratios": [1.0, 1.0, 0.06], "wspace": 0.18})
        cbar_ax = axes[2]
        for pi, mt in enumerate(mtypes):
            ax = axes[pi]
            mat = mat_by_type[mt]
            im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(n_rates))
            ax.set_xticklabels([f"{r}%" for r in rates], fontsize=TICK_FS - 0.5)
            ax.set_yticks(range(len(ABL_MODEL_ORDER)))
            ax.set_yticklabels([MODEL_NAMES[m] for m in ABL_MODEL_ORDER]
                                if pi == 0 else [], fontsize=TICK_FS - 0.5)
            ax.set_xlabel("Masking rate")
            for i in range(len(ABL_MODEL_ORDER)):
                for j in range(n_rates):
                    v = mat[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                 fontsize=TICK_FS - 1.5,
                                 color="white" if (v < 0.3 or v > 0.78)
                                 else "#222")
            ax.set_title(f"({chr(97 + pi)}) {mt.capitalize()} masking",
                          pad=4, loc="center")
            ax.grid(False)

        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label(r"Macro-F$_1$", fontsize=LBL_FS)
        cbar.ax.tick_params(labelsize=TICK_FS - 0.5)
        save_fig(fig, out_path)


def fig_mask_perclass(out_path: Path):
    metrics = _load_ablation(ABL_MASK / "metrics" / "mask_all_metrics.json")
    rates = [10, 20, 40, 60, 80]
    mtypes = ["word", "utterance"]
    classes = LABELS_3CLASS

    with mpl.rc_context(AIAA_RC):
        fig, axes = plt.subplots(2, len(ABL_MODEL_ORDER),
                                  figsize=(PAGE_W, HALF_W * 1.6),
                                  sharex=True, sharey=True)
        for ri, mt in enumerate(mtypes):
            for ci, mk in enumerate(ABL_MODEL_ORDER):
                ax = axes[ri][ci]
                for c in classes:
                    ys = []
                    for r in rates:
                        key = f"{mk}_zero_shot_mask_{mt}_{r}"
                        f = metrics.get(key, {}).get("per_class", {}) \
                                    .get(c, {}).get("f1")
                        ys.append(f if f is not None else np.nan)
                    ax.plot(rates, ys,
                             color=CLASS_COLORS[c],
                             marker={"nominal":"o","warning":"s","hazard":"^"}[c],
                             markersize=5, linewidth=1.2,
                             label=LABEL_NAMES[c])
                ax.set_xticks(rates)
                ax.set_ylim(0.0, 1.0)
                ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
                ax.grid(axis="y", color="#DDD", linewidth=0.4,
                         linestyle="--", zorder=0)
                if ri == 0:
                    ax.set_title(MODEL_NAMES[mk], fontsize=LBL_FS,
                                 pad=4, loc="center")
                if ri == 1:
                    ax.set_xlabel("Masking rate, %")
                if ci == 0:
                    ax.set_ylabel(f"{mt.capitalize()} masking\n"
                                   r"Per-class F$_1$",
                                   fontsize=LBL_FS - 0.5)
                if ri == 0 and ci == len(ABL_MODEL_ORDER) - 1:
                    ax.legend(loc="lower left", frameon=True, framealpha=0.92,
                               edgecolor="#CCC", fontsize=LEG_FS - 1,
                               title="Class")
        fig.tight_layout(pad=0.4)
        save_fig(fig, out_path)


def _fmt(v, best, decimals=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return r"--"
    s = f"{v:.{decimals}f}"
    return r"\textbf{" + s + "}" if (best is not None and abs(v - best) < 1e-9) else s


def _auroc_for_run(raw_dir: Path, run_key_str: str, positive_class: str):
    f = raw_dir / f"{run_key_str}.json"
    if not f.exists():
        return None
    records = json.load(open(f)).get("records", [])
    y_true, y_scores = _scores_from_records(records, positive_class)
    if not y_true or sum(y_true) in (0, len(y_true)):
        return None
    _, _, auc = _roc_curve(y_true, y_scores)
    return auc


def write_perclass_table(metrics, classes, out_path: Path,
                          caption: str = "",
                          label: str = "tab:perclass"):
    """Per-class F1 hierarchical table. Rows: (model, class). Cols: 6 strategies"""

    def cell_value(model, strat, cot, cls):
        rk = run_key(model, strat, cot)
        m = metrics.get(rk)
        if not m:
            return None
        return m.get("per_class", {}).get(cls, {}).get("f1")

    def best_for_row(model, cls):
        vals = []
        for cot in [False, True]:
            for s in STRATEGIES:
                v = cell_value(model, s, cot, cls)
                if v is not None:
                    vals.append(v)
        return max(vals) if vals else None

    K = len(classes)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\small",
        r"\begin{tabular}{l l l *{6}{c}}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{Direct prompting} "
        r"& \multicolumn{3}{c}{Chain-of-thought} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"Source & Model & Class & ZS & OS & FS & ZS & OS & FS \\",
        r"\midrule",
    ]

    sources = [("Open-source", [m for m in MODEL_ORDER if m in OPEN_SOURCE]),
               ("Closed-source", [m for m in MODEL_ORDER if m in CLOSED_SOURCE])]

    for source_idx, (source_name, models_in_src) in enumerate(sources):
        rows_in_block = len(models_in_src) * K
        first = True
        for mi, m in enumerate(models_in_src):
            for ki, cls in enumerate(classes):
                cells = []
                row_best = best_for_row(m, cls)
                for cot in [False, True]:
                    for s in STRATEGIES:
                        cells.append(_fmt(cell_value(m, s, cot, cls),
                                           row_best))
                src_cell = (rf"\multirow{{{rows_in_block}}}{{*}}{{\textit{{{source_name}}}}}"
                            if first else "")
                first = False
                model_cell = (rf"\multirow{{{K}}}{{*}}{{{MODEL_NAMES[m]}}}"
                              if ki == 0 else "")
                lines.append(
                    f"  {src_cell} & {model_cell} & {LABEL_NAMES[cls]} & "
                    + " & ".join(cells) + r" \\"
                )
            if mi < len(models_in_src) - 1:
                lines.append(rf"\cmidrule(lr){{2-9}}")
        if source_idx < len(sources) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved -> {out_path.relative_to(OUT)}")


def write_main_results_table(metrics, classes, out_path: Path,
                              raw_dir: Path = None,
                              include_auroc: bool = False,
                              caption: str = "",
                              label: str = "tab:main"):
    """Hierarchical table: Source x Method x Strategy. Rows: (model, metric)."""
    metric_names = ["Macro-F$_1$", "Accuracy"]
    metric_keys  = ["macro_f1",    "accuracy"]
    if include_auroc:
        metric_names.append("AUROC")
        metric_keys.append("auroc")

    # Pre-compute AUROC per run if requested
    auroc_cache = {}
    if include_auroc and raw_dir is not None:
        for rk in metrics:
            auroc_cache[rk] = _auroc_for_run(raw_dir, rk, "danger")

    def cell_value(model, strat, cot, metric_key):
        rk = run_key(model, strat, cot)
        m = metrics.get(rk)
        if not m:
            return None
        if metric_key == "auroc":
            return auroc_cache.get(rk)
        return m.get(metric_key)

    # For each (model, metric_key) row, find best across strategies × cot
    def best_for_row(model, metric_key):
        vals = []
        for cot in [False, True]:
            for s in STRATEGIES:
                v = cell_value(model, s, cot, metric_key)
                if v is not None:
                    vals.append(v)
        return max(vals) if vals else None

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\small",
        # 3 leading cols + 6 numeric cols
        r"\begin{tabular}{l l l *{6}{c}}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{Direct prompting} "
        r"& \multicolumn{3}{c}{Chain-of-thought} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"Source & Model & Metric & ZS & OS & FS & ZS & OS & FS \\",
        r"\midrule",
    ]

    sources = [("Open-source", [m for m in MODEL_ORDER if m in OPEN_SOURCE]),
               ("Closed-source", [m for m in MODEL_ORDER if m in CLOSED_SOURCE])]

    for source_idx, (source_name, models_in_src) in enumerate(sources):
        n_metric = len(metric_keys)
        rows_in_block = len(models_in_src) * n_metric
        first = True
        for mi, m in enumerate(models_in_src):
            for ki, (mn, mk) in enumerate(zip(metric_names, metric_keys)):
                cells = []
                row_best = best_for_row(m, mk)
                for cot in [False, True]:
                    for s in STRATEGIES:
                        cells.append(_fmt(cell_value(m, s, cot, mk),
                                           row_best))

                if first:
                    src_cell = rf"\multirow{{{rows_in_block}}}{{*}}{{\textit{{{source_name}}}}}"
                    first = False
                else:
                    src_cell = ""

                if ki == 0:
                    model_cell = rf"\multirow{{{n_metric}}}{{*}}{{{MODEL_NAMES[m]}}}"
                else:
                    model_cell = ""

                lines.append(
                    f"  {src_cell} & {model_cell} & {mn} & "
                    + " & ".join(cells) + r" \\"
                )
            if mi < len(models_in_src) - 1:
                lines.append(rf"\cmidrule(lr){{2-9}}")
        if source_idx < len(sources) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved -> {out_path.relative_to(OUT)}")


def copy_architecture():
    """Re-run make_architecture.py to refresh the figure with current style,"""
    src = Path("figures") / "architecture.pdf"
    if src.exists():
        dst = OUT_FIG / "architecture.pdf"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  saved -> {dst.relative_to(OUT)}")


def main():
    OUT.mkdir(exist_ok=True)
    OUT_FIG_M3.mkdir(parents=True, exist_ok=True)
    OUT_FIG_MB.mkdir(parents=True, exist_ok=True)
    OUT_FIG_AB.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)

    print("Rebuilding 3-class metrics from results/raw/")
    m3, ph3 = rebuild_main_metrics(RAW_3CLASS, LABELS_3CLASS)
    print(f"  -> {len(m3)} runs")

    print("Rebuilding binary metrics from results_binary/raw/")
    mb, phb = rebuild_main_metrics(RAW_BINARY, LABELS_BINARY)
    print(f"  -> {len(mb)} runs")

    print("\nArchitecture figure")
    copy_architecture()

    print("\nMain 3-class figures (paper_assets/figures/main_3class/)")
    fig_strategy_lines  (m3, LABELS_3CLASS, OUT_FIG_M3 / "strategy_improvement.pdf")
    fig_per_hazard      (ph3, m3, LABELS_3CLASS, OUT_FIG_M3 / "per_hazard_accuracy.pdf")
    fig_latency_two_panel(m3, OUT_FIG_M3 / "latency_comparison.pdf")
    fig_cot_delta       (m3, OUT_FIG_M3 / "cot_improvement.pdf")
    fig_confusion_matrix_grid(m3, LABELS_3CLASS,
                               OUT_FIG_M3 / "confusion_matrix_grid.pdf")

    print("\nMain binary figures (paper_assets/figures/main_binary/)")
    fig_strategy_lines  (mb, LABELS_BINARY, OUT_FIG_MB / "strategy_improvement.pdf")
    fig_per_hazard      (phb, mb, LABELS_BINARY, OUT_FIG_MB / "per_hazard_accuracy.pdf")
    fig_cot_delta       (mb, OUT_FIG_MB / "cot_improvement.pdf")
    fig_confusion_matrix_grid(mb, LABELS_BINARY,
                               OUT_FIG_MB / "confusion_matrix_grid.pdf")
    fig_pr_roc(RAW_BINARY, mb, OUT_FIG_MB / "pr_roc_curves.pdf",
                positive_class="danger")

    print("\nAblation figures (paper_assets/figures/ablations/)")
    fig_whisper_compare    (OUT_FIG_AB / "fig_asr_comparison.pdf")
    fig_whisper_perclass   (OUT_FIG_AB / "fig_asr_perclass_f1.pdf")
    fig_noise_curve        (OUT_FIG_AB / "fig_noise_f1_curve.pdf")
    fig_noise_perclass     (OUT_FIG_AB / "fig_noise_perclass.pdf")
    fig_mask_curve         (OUT_FIG_AB / "fig_mask_f1_curve.pdf")
    fig_mask_heatmap       (OUT_FIG_AB / "fig_mask_heatmap.pdf")
    fig_mask_perclass      (OUT_FIG_AB / "fig_mask_perclass.pdf")

    print("\nTables (paper_assets/tables/)")
    write_main_results_table(
        m3, LABELS_3CLASS, OUT_TAB / "table_main_3class.tex",
        include_auroc=False,
        caption=("Three-class classification results (Nominal / Warning / "
                 "Hazard) on the CTAF-KHAF benchmark. Best per (model, "
                 "metric) row in \\textbf{bold}."),
        label="tab:main_3class")
    write_main_results_table(
        mb, LABELS_BINARY, OUT_TAB / "table_main_binary.tex",
        raw_dir=RAW_BINARY,
        include_auroc=True,
        caption=("Binary classification results (Nominal vs.\\ Danger) on "
                 "the CTAF-KHAF benchmark. Best per (model, metric) row "
                 "in \\textbf{bold}. AUROC is computed against the danger "
                 "class using token logprobs where available, with "
                 "self-reported confidence as a fallback for models whose "
                 "API does not expose logprobs."),
        label="tab:main_binary")
    write_perclass_table(
        m3, LABELS_3CLASS, OUT_TAB / "table_perclass_f1_3class.tex",
        caption=("Per-class F$_1$ scores on the three-class CTAF-KHAF "
                 "benchmark across all models, prompting strategies, and "
                 "reasoning methods. Best per (model, class) row in "
                 "\\textbf{bold}."),
        label="tab:perclass_3class")
    write_perclass_table(
        mb, LABELS_BINARY, OUT_TAB / "table_perclass_f1_binary.tex",
        caption=("Per-class F$_1$ scores on the binary CTAF-KHAF "
                 "benchmark (Nominal vs.\\ Danger) across all models, "
                 "prompting strategies, and reasoning methods. Best per "
                 "(model, class) row in \\textbf{bold}."),
        label="tab:perclass_binary")

    print("\nDone. paper_assets/ tree:")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT)}")


if __name__ == "__main__":
    main()
