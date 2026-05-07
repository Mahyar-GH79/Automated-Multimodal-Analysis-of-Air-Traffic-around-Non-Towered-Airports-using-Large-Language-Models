#!/usr/bin/env python3
"""ICML-style figures + tables for the CTAF-KHAF benchmark."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D


COLUMN_W   = 3.5
PAGE_W     = 7.0
FONT_SIZE  = 8.5
TITLE_SIZE = 9.5
TICK_SIZE  = 7.5
LEGEND_SIZE = 7.5
DPI        = 300
LINE_W     = 1.2
MARKER_S   = 5

C_NOM  = "#4DAC26"
C_WARN = "#E08214"
C_HAZ  = "#D6604D"

# Per-model colors (5 models)
MODEL_COLORS = {
    "qwen":              "#2166AC",
    "mistral":           "#762A83",
    "gemma":             "#1B7837",
    "gpt-4o":            "#B35806",
    "gpt-5.4":           "#000000",
    "claude-sonnet-4-6": "#CC4F1B",
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

# Canonical order for plots (left-to-right)
MODEL_ORDER = ["qwen", "mistral", "gemma", "gpt-4o", "gpt-5.4",
               "claude-sonnet-4-6"]

STRATEGIES = ["zero_shot", "one_shot", "few_shot"]
STRAT_NAMES = {"zero_shot": "Zero-shot", "one_shot": "One-shot", "few_shot": "Few-shot"}
STRAT_COLORS = {"zero_shot": "#4393C3", "one_shot": "#F4A582", "few_shot": "#1B7837"}

LABELS = ["nominal", "warning", "hazard"]
LABEL_NAMES  = {"nominal": "Nominal", "warning": "Warning", "hazard": "Hazard",
                "danger": "Danger"}
LABEL_COLORS = {"nominal": C_NOM, "warning": C_WARN, "hazard": C_HAZ,
                "danger": C_HAZ}

LABELS_BINARY = ["nominal", "danger"]


def detect_binary_mode(raw_dir: Path) -> bool:
    """Return True if any raw record uses 'danger' as ground truth."""
    for f in raw_dir.glob("*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in d.get("records", [])[:5]:
            if r.get("ground_truth") == "danger" or r.get("predicted") == "danger":
                return True
    return False

HAZARD_LABEL = {
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
    "imc_vfr_conflict": "IMC/VFR conflict",
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


def icml_style():
    return {
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset":   "dejavuserif",
        "font.size":          FONT_SIZE,
        "axes.titlesize":     TITLE_SIZE,
        "axes.labelsize":     FONT_SIZE,
        "xtick.labelsize":    TICK_SIZE,
        "ytick.labelsize":    TICK_SIZE,
        "legend.fontsize":    LEGEND_SIZE,
        "legend.title_fontsize": LEGEND_SIZE,
        "lines.linewidth":    LINE_W,
        "lines.markersize":   MARKER_S,
        "patch.linewidth":    0.5,
        "axes.linewidth":     0.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,
        "axes.axisbelow":     True,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "figure.dpi":         DPI,
        "savefig.dpi":        DPI,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    }


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path.name}")


def parse_run_key(run_key: str):
    """qwen_zero_shot_cot -> ('qwen','zero_shot',True). gpt-4o_few_shot -> ('gpt-4o','few_shot',False)."""
    cot = run_key.endswith("_cot")
    body = run_key[:-4] if cot else run_key
    for st in STRATEGIES:
        suf = "_" + st
        if body.endswith(suf):
            model = body[:-len(suf)]
            return model, st, cot
    raise ValueError(f"cannot parse run_key={run_key!r}")


def run_key(model, strategy, cot):
    return f"{model}_{strategy}{'_cot' if cot else ''}"


def compute_run_metrics(records: list[dict]) -> dict:
    """Compute per-run metrics from a list of record dicts."""
    classes = LABELS
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    n = 0
    n_correct = 0
    n_parse_ok = 0
    latencies = []
    for r in records:
        gt = r.get("ground_truth")
        pred = r.get("predicted")
        if gt not in classes or pred not in classes:
            # Treat unparseable predictions as "wrong" by mapping to the worst-case
            # label for the GT - but for confusion-matrix purposes, skip them.
            n += 1
            if r.get("parse_ok"):
                n_parse_ok += 1
            if r.get("latency_s") is not None:
                latencies.append(r["latency_s"])
            continue
        i = classes.index(gt)
        j = classes.index(pred)
        cm[i, j] += 1
        n += 1
        if i == j:
            n_correct += 1
        if r.get("parse_ok"):
            n_parse_ok += 1
        if r.get("latency_s") is not None:
            latencies.append(r["latency_s"])

    accuracy = n_correct / n if n else 0.0
    parse_ok = n_parse_ok / n if n else 0.0
    avg_lat = float(np.mean(latencies)) if latencies else 0.0

    per_class = {}
    f1s = []
    precs = []
    recs = []
    for k, c in enumerate(classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": int(cm[k, :].sum())}
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

    return {
        "accuracy": accuracy,
        "macro_p": float(np.mean(precs)),
        "macro_r": float(np.mean(recs)),
        "macro_f1": float(np.mean(f1s)),
        "parse_ok": parse_ok,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n": n,
        "avg_latency_s": avg_lat,
    }


def compute_per_hazard(records: list[dict]) -> dict:
    """For each hazard_type: accuracy and counts."""
    out = {}
    by_ht = defaultdict(list)
    for r in records:
        ht = r.get("hazard_type")
        if ht:
            by_ht[ht].append(r)
    for ht, rs in by_ht.items():
        n = len(rs)
        n_correct = sum(1 for r in rs if r.get("ground_truth") == r.get("predicted"))
        out[ht] = {"accuracy": n_correct / n if n else 0.0, "n": n}
    return out


def rebuild_all_metrics(raw_dir: Path):
    """Load every results/raw/*.json and compute fresh metrics."""
    metrics = {}
    per_hazard = {}
    for f in sorted(raw_dir.glob("*.json")):
        rk = f.stem
        try:
            data = json.load(open(f))
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")
            continue
        records = data.get("records", [])
        if not records:
            print(f"  [skip] {f.name}: no records")
            continue
        try:
            model, strat, cot = parse_run_key(rk)
        except ValueError:
            print(f"  [skip] {f.name}: cannot parse run key")
            continue
        m = compute_run_metrics(records)
        m["model_key"] = model
        m["strategy"] = strat
        m["cot"] = cot
        metrics[rk] = m
        per_hazard[rk] = compute_per_hazard(records)
    return metrics, per_hazard


def fig_macro_f1_panels(metrics, out_dir):
    """Two side-by-side panels: Direct (no-CoT) and CoT. Grouped bars by strategy."""
    fig_path = out_dir / "f1_by_model_strategy.pdf"

    with mpl.rc_context(icml_style()):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, COLUMN_W * 1.0),
                                  sharey=True)
        for pi, cot_flag in enumerate([False, True]):
            ax = axes[pi]
            present_models = [m for m in MODEL_ORDER
                              if any(run_key(m, s, cot_flag) in metrics
                                     for s in STRATEGIES)]
            if not present_models:
                ax.set_visible(False)
                continue
            x = np.arange(len(present_models))
            bw = 0.26
            for si, st in enumerate(STRATEGIES):
                vals = [metrics.get(run_key(m, st, cot_flag), {}).get("macro_f1", np.nan)
                        for m in present_models]
                xpos = x + (si - 1) * bw
                bars = ax.bar(xpos, vals, width=bw,
                               color=STRAT_COLORS[st],
                               edgecolor="white", linewidth=0.5,
                               label=STRAT_NAMES[st], zorder=3)
                for b, v in zip(bars, vals):
                    if not np.isnan(v) and v > 0.04:
                        ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                                f"{v:.2f}", ha="center", va="bottom",
                                fontsize=5.8, color="#222")
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_NAMES[m] for m in present_models],
                                fontsize=TICK_SIZE - 0.5, rotation=20, ha="right")
            ax.set_ylim(0, 0.95)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.5,
                     linestyle="--", zorder=0)
            title = "(a) Direct prompting" if not cot_flag else "(b) Chain-of-thought"
            ax.set_title(title, pad=4, loc="left")
            if pi == 0:
                ax.set_ylabel("Macro-averaged F$_1$")
            if pi == 1:
                ax.legend(loc="upper left", frameon=True, framealpha=0.92,
                           edgecolor="#CCCCCC", title="Prompting strategy",
                           fontsize=LEGEND_SIZE - 0.5,
                           title_fontsize=LEGEND_SIZE - 0.5)
        fig.tight_layout(pad=0.4)
        save_fig(fig, fig_path)


def fig_strategy_lines(metrics, out_dir):
    """ICL line plot: x=strategy, y=F1, one line per model. Two panels (direct, CoT)."""
    fig_path = out_dir / "strategy_improvement.pdf"

    with mpl.rc_context(icml_style()):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, COLUMN_W * 1.0),
                                  sharey=True)
        for pi, cot_flag in enumerate([False, True]):
            ax = axes[pi]
            x = np.arange(len(STRATEGIES))
            for m in MODEL_ORDER:
                ys = [metrics.get(run_key(m, s, cot_flag), {}).get("macro_f1", None)
                      for s in STRATEGIES]
                if not any(v is not None for v in ys):
                    continue
                yarr = [v if v is not None else np.nan for v in ys]
                ax.plot(x, yarr,
                         color=MODEL_COLORS[m],
                         marker=MODEL_MARKERS[m],
                         markersize=MARKER_S + 1,
                         linewidth=LINE_W + 0.3,
                         label=MODEL_NAMES[m], zorder=4)
                for xi, yi in zip(x, yarr):
                    if not np.isnan(yi):
                        ax.annotate(f"{yi:.2f}",
                                    xy=(xi, yi), xytext=(0, 7),
                                    textcoords="offset points",
                                    ha="center", fontsize=6,
                                    color=MODEL_COLORS[m])
            ax.set_xticks(x)
            ax.set_xticklabels([STRAT_NAMES[s] for s in STRATEGIES])
            ax.set_ylim(0.30, 0.95)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.4,
                     linestyle="--", zorder=0)
            title = "(a) Direct prompting" if not cot_flag else "(b) Chain-of-thought"
            ax.set_title(title, pad=4, loc="left")
            if pi == 0:
                ax.set_ylabel("Macro-averaged F$_1$")
            if pi == 1:
                ax.legend(loc="lower right", frameon=True,
                           framealpha=0.92, edgecolor="#CCCCCC",
                           fontsize=LEGEND_SIZE - 0.5)
        fig.tight_layout(pad=0.4)
        save_fig(fig, fig_path)


def fig_perclass_heatmap(metrics, out_dir):
    """Single heatmap: rows=class, cols=run (model x strategy x cot)."""
    fig_path = out_dir / "per_class_f1_heatmap.pdf"

    cols = []
    col_labs = []
    col_groups = []
    data = {l: [] for l in LABELS}
    for m in MODEL_ORDER:
        for cot_flag in [False, True]:
            for s in STRATEGIES:
                k = run_key(m, s, cot_flag)
                if k not in metrics:
                    continue
                cols.append(k)
                strat_abbr = {"zero_shot": "ZS", "one_shot": "OS", "few_shot": "FS"}[s]
                col_labs.append(strat_abbr + ("+CoT" if cot_flag else ""))
                col_groups.append(m)
                pc = metrics[k]["per_class"]
                for l in LABELS:
                    data[l].append(pc[l]["f1"])

    if not cols:
        print("  per_class_f1_heatmap: no data")
        return
    mat = np.array([data[l] for l in LABELS])

    with mpl.rc_context(icml_style()):
        n_cols = len(cols)
        fig, ax = plt.subplots(figsize=(max(PAGE_W, n_cols * 0.32),
                                         COLUMN_W * 0.95))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
        cbar.ax.tick_params(labelsize=TICK_SIZE - 0.5)
        cbar.set_label("F$_1$", fontsize=FONT_SIZE - 0.5)
        ax.set_yticks(range(len(LABELS)))
        ax.set_yticklabels([LABEL_NAMES[l] for l in LABELS])
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labs, fontsize=TICK_SIZE - 1.5,
                            rotation=70, ha="right")
        ax.grid(False)

        # Vertical separators between models
        seps = []
        for i in range(1, n_cols):
            if col_groups[i] != col_groups[i - 1]:
                seps.append(i - 0.5)
        for s in seps:
            ax.axvline(s, color="#888", linewidth=0.7, linestyle="--")

        # Model headers
        unique_models, group_centers = [], []
        i = 0
        while i < n_cols:
            j = i
            while j < n_cols and col_groups[j] == col_groups[i]:
                j += 1
            unique_models.append(col_groups[i])
            group_centers.append((i + j - 1) / 2)
            i = j
        for cx, mk in zip(group_centers, unique_models):
            ax.text(cx, -0.6, MODEL_NAMES[mk], ha="center", va="bottom",
                    fontsize=FONT_SIZE - 0.5, fontweight="bold",
                    color=MODEL_COLORS[mk],
                    transform=ax.get_xaxis_transform())

        for i in range(len(LABELS)):
            for j in range(n_cols):
                v = mat[i, j]
                tc = "white" if (v < 0.25 or v > 0.78) else "#222"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color=tc)
        ax.set_title("Per-class F$_1$ across all runs", pad=14, loc="left")
        fig.tight_layout(pad=0.4)
        save_fig(fig, fig_path)


def fig_per_hazard(per_hazard, metrics, out_dir):
    """Per-hazard accuracy. Pick best CoT run per model + GPT-4o best non-CoT run."""
    fig_path = out_dir / "per_hazard_accuracy.pdf"

    # Best run per model: highest macro F1
    best = {}
    for m in MODEL_ORDER:
        candidates = [(k, v) for k, v in metrics.items()
                       if v["model_key"] == m]
        if not candidates:
            continue
        best[m] = max(candidates, key=lambda kv: kv[1]["macro_f1"])[0]

    if not best:
        print("  per_hazard: no data")
        return

    ht_order = sorted(HAZARD_LABEL.keys(),
                       key=lambda ht: ({"hazard": 0, "warning": 1, "nominal": 2}[HAZARD_LABEL[ht]],
                                        ht))
    with mpl.rc_context(icml_style()):
        fig, ax = plt.subplots(figsize=(PAGE_W, COLUMN_W * 2.4))
        models_present = [m for m in MODEL_ORDER if m in best]
        n_models = len(models_present)
        bar_h = 0.13
        offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_h * 1.05

        # Y positions with gaps between class groups
        y_pos = []
        cur_y = 0.0
        prev_cls = None
        for ht in ht_order:
            cls = HAZARD_LABEL[ht]
            if prev_cls is not None and cls != prev_cls:
                cur_y += 0.5
            y_pos.append(cur_y)
            cur_y += 0.85
            prev_cls = cls
        y_pos = np.array(y_pos)

        for mi, m in enumerate(models_present):
            rk = best[m]
            ph = per_hazard.get(rk, {})
            vals = [ph.get(ht, {}).get("accuracy", np.nan) for ht in ht_order]
            ax.barh(y_pos + offsets[mi], vals,
                     height=bar_h * 0.95,
                     color=MODEL_COLORS[m],
                     edgecolor="white", linewidth=0.3,
                     label=f"{MODEL_NAMES[m]} ({STRAT_NAMES[parse_run_key(rk)[1]][0]}S"
                           + ("+CoT" if parse_run_key(rk)[2] else "") + ")",
                     zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([HT_DISPLAY[ht] for ht in ht_order],
                            fontsize=TICK_SIZE - 0.5)
        for tick, ht in zip(ax.get_yticklabels(), ht_order):
            tick.set_color(LABEL_COLORS[HAZARD_LABEL[ht]])
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.set_xlabel("Accuracy")
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.5,
                 linestyle="--", zorder=0)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92,
                   edgecolor="#CCCCCC", fontsize=LEGEND_SIZE - 1,
                   title="Best run per model")
        ax.set_title("Per-hazard-type accuracy", pad=5, loc="left")
        fig.tight_layout(pad=0.4)
        save_fig(fig, fig_path)


def fig_latency(metrics, out_dir):
    """Two-panel: avg latency per model + F1 vs latency scatter."""
    fig_path = out_dir / "latency_comparison.pdf"

    with mpl.rc_context(icml_style()):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, COLUMN_W * 1.0))

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
                      fontsize=7, color="#222")
        ax0.set_ylabel("Avg. inference latency (s / scenario)")
        ax0.set_ylim(0, max(avg_lat) * 1.25 if avg_lat else 1.0)
        ax0.tick_params(axis="x", labelrotation=20)
        ax0.grid(axis="y", color="#DDDDDD", linewidth=0.5,
                  linestyle="--", zorder=0)
        ax0.set_title("(a) Avg. latency per model", pad=4, loc="left")

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
                              markersize=7,
                              linestyle="none",
                              markeredgewidth=1.0, zorder=5)
        ax1.set_xlabel("Avg. latency (s / scenario)")
        ax1.set_ylabel("Macro-averaged F$_1$")
        ax1.grid(color="#DDDDDD", linewidth=0.4, linestyle="--", zorder=0)

        # Compact legend
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
        ax1.set_title("(b) F$_1$ vs.\\ latency trade-off", pad=4, loc="left")
        fig.tight_layout(pad=0.4)
        save_fig(fig, fig_path)


def fig_cot_delta(metrics, out_dir):
    """Bar chart: F1 improvement from adding CoT, per (model, strategy)."""
    fig_path = out_dir / "cot_improvement.pdf"

    rows = []  # (model, strategy, direct_f1, cot_f1, delta)
    for m in MODEL_ORDER:
        for s in STRATEGIES:
            d = metrics.get(run_key(m, s, False), {}).get("macro_f1")
            c = metrics.get(run_key(m, s, True), {}).get("macro_f1")
            if d is not None and c is not None:
                rows.append((m, s, d, c, c - d))
    if not rows:
        print("  cot_improvement: no paired runs")
        return

    with mpl.rc_context(icml_style()):
        fig, ax = plt.subplots(figsize=(PAGE_W, COLUMN_W * 0.95))
        x = np.arange(len(rows))
        deltas = [r[4] for r in rows]
        colors = [MODEL_COLORS[r[0]] for r in rows]
        bars = ax.bar(x, deltas, width=0.7, color=colors,
                       edgecolor="white", linewidth=0.5, zorder=3)
        for b, d in zip(bars, deltas):
            offs = 0.008 if d >= 0 else -0.018
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + offs,
                    f"{d:+.2f}", ha="center",
                    va="bottom" if d >= 0 else "top",
                    fontsize=6, color="#222")
        ax.axhline(0, color="#444", linewidth=0.7)
        labs = [f"{r[1].split('_')[0][0].upper()}S" for r in rows]
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=TICK_SIZE - 0.5)

        # Model group headers
        cur = 0
        i = 0
        while i < len(rows):
            j = i
            while j < len(rows) and rows[j][0] == rows[i][0]:
                j += 1
            cx = (i + j - 1) / 2
            ax.text(cx, -0.18, MODEL_NAMES[rows[i][0]],
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=FONT_SIZE - 0.5,
                    color=MODEL_COLORS[rows[i][0]], fontweight="bold")
            if j < len(rows):
                ax.axvline(j - 0.5, color="#BBB", linewidth=0.7,
                            linestyle="--")
            i = j

        ax.set_ylabel(r"$\Delta$ Macro-F$_1$ (CoT $-$ Direct)")
        ymin = min(deltas) - 0.05
        ymax = max(deltas) + 0.05
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.2f"))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.4,
                 linestyle="--", zorder=0)
        ax.set_title("Effect of chain-of-thought prompting on F$_1$",
                      pad=18, loc="left")
        fig.tight_layout(pad=0.5)
        save_fig(fig, fig_path)


def fig_confusion_matrices(metrics, out_dir):
    """One PDF per run: KxK confusion matrix (K = number of classes)."""
    LABELS_CAP = [LABEL_NAMES[l] for l in LABELS]
    K = len(LABELS)
    for rk, m in metrics.items():
        cm = np.array(m["confusion_matrix"], dtype=float)
        rs = cm.sum(axis=1, keepdims=True).clip(min=1)
        cmn = cm / rs
        with mpl.rc_context(icml_style()):
            fig, ax = plt.subplots(figsize=(COLUMN_W * 1.3, COLUMN_W * 1.2))
            im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for i in range(K):
                for j in range(K):
                    v = cmn[i, j]
                    cnt = int(cm[i, j])
                    bold = (i == j)
                    tc = "white" if v > 0.6 else "#1a1a1a"
                    ax.text(j, i, f"{cnt}\n({v:.0%})", ha="center", va="center",
                             fontsize=8, color=tc,
                             fontweight="bold" if bold else "normal")
            ax.set_xticks(range(K))
            ax.set_yticks(range(K))
            ax.set_xticklabels(LABELS_CAP, fontsize=TICK_SIZE)
            ax.set_yticklabels(LABELS_CAP, fontsize=TICK_SIZE)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.grid(False)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            model, strat, cot = parse_run_key(rk)
            title = f"{MODEL_NAMES.get(model, model)} -- {STRAT_NAMES[strat]}"
            if cot:
                title += " + CoT"
            ax.set_title(title, pad=6, loc="left")
            fig.tight_layout(pad=0.3)
            save_fig(fig, out_dir / f"cm_{rk}.pdf")


def _scores_from_records(records: list[dict], positive_class: str) -> tuple[list, list]:
    """Return (y_true_binary, y_scores_for_positive_class)."""
    y_true = []
    y_scores = []
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
            y_scores.append(conf if r.get("predicted") == positive_class else 1.0 - conf)
    return y_true, y_scores


def _pr_curve(y_true, y_scores):
    """Precision-recall curve (no sklearn). Returns (precision, recall, thresholds, ap)."""
    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    P = sum(y_true) or 1
    tp = fp = 0
    precision = [1.0]
    recall = [0.0]
    thr = []
    last_score = None
    for s, y in pairs:
        if last_score is not None and s != last_score:
            precision.append(tp / max(tp + fp, 1))
            recall.append(tp / P)
            thr.append(last_score)
        if y == 1:
            tp += 1
        else:
            fp += 1
        last_score = s
    precision.append(tp / max(tp + fp, 1))
    recall.append(tp / P)
    thr.append(last_score if last_score is not None else 0.0)
    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i - 1]) * precision[i]
    return precision, recall, thr, ap


def _roc_curve(y_true, y_scores):
    """ROC curve (no sklearn). Returns (fpr, tpr, thresholds, auc)."""
    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    P = sum(y_true) or 1
    N = len(y_true) - sum(y_true) or 1
    tp = fp = 0
    fpr = [0.0]
    tpr = [0.0]
    thr = []
    last_score = None
    for s, y in pairs:
        if last_score is not None and s != last_score:
            fpr.append(fp / N)
            tpr.append(tp / P)
            thr.append(last_score)
        if y == 1:
            tp += 1
        else:
            fp += 1
        last_score = s
    fpr.append(fp / N)
    tpr.append(tp / P)
    thr.append(last_score if last_score is not None else 0.0)
    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return fpr, tpr, thr, auc


def fig_pr_roc(raw_dir: Path, metrics: dict, out_dir: Path,
                positive_class: str = "danger"):
    """Two-panel figure: PR (left) and ROC (right). One curve per model = best"""
    fig_path = out_dir / "pr_roc_curves.pdf"

    # Best run per model by macro-F1
    best = {}
    for rk, m in metrics.items():
        mk = m["model_key"]
        if mk not in best or m["macro_f1"] > best[mk][1]["macro_f1"]:
            best[mk] = (rk, m)

    if not best:
        print("  pr_roc: no metrics")
        return

    curves = []
    for mk in MODEL_ORDER:
        if mk not in best:
            continue
        rk, m = best[mk]
        raw_path = raw_dir / f"{rk}.json"
        if not raw_path.exists():
            continue
        records = json.load(open(raw_path)).get("records", [])
        if not records:
            continue
        y_true, y_scores = _scores_from_records(records, positive_class)
        if not y_true or sum(y_true) == 0 or sum(y_true) == len(y_true):
            continue  # PR/ROC undefined
        prec, rec, _, ap = _pr_curve(y_true, y_scores)
        fpr, tpr, _, auc = _roc_curve(y_true, y_scores)
        sample = records[: min(5, len(records))]
        score_src = ("confidence" if any(
                         r.get("score_source") == "confidence_fallback"
                         for r in sample)
                     else "logprobs")
        curves.append((mk, rk, prec, rec, ap, fpr, tpr, auc, score_src))

    if not curves:
        print("  pr_roc: nothing to plot")
        return

    with mpl.rc_context(icml_style()):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, COLUMN_W * 1.05))

        ax = axes[0]
        for mk, rk, prec, rec, ap, fpr, tpr, auc, src in curves:
            label = f"{MODEL_NAMES[mk]} (AP={ap:.3f}"
            if src == "confidence":
                label += ", conf*)"
            else:
                label += ")"
            ax.plot(rec, prec,
                     color=MODEL_COLORS[mk],
                     marker=MODEL_MARKERS[mk], markersize=3,
                     markevery=max(len(rec) // 10, 1),
                     linewidth=LINE_W + 0.2,
                     label=label)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#DDD", linewidth=0.4, linestyle="--", zorder=0)
        ax.set_title(f"(a) PR curve, positive = {positive_class}",
                      pad=4, loc="left")
        ax.legend(loc="lower left", frameon=True, framealpha=0.92,
                   edgecolor="#CCC", fontsize=LEGEND_SIZE - 1)

        ax = axes[1]
        ax.plot([0, 1], [0, 1], color="#888", linestyle=":",
                 linewidth=0.8, zorder=1)
        for mk, rk, prec, rec, ap, fpr, tpr, auc, src in curves:
            label = f"{MODEL_NAMES[mk]} (AUC={auc:.3f}"
            if src == "confidence":
                label += ", conf*)"
            else:
                label += ")"
            ax.plot(fpr, tpr,
                     color=MODEL_COLORS[mk],
                     marker=MODEL_MARKERS[mk], markersize=3,
                     markevery=max(len(fpr) // 10, 1),
                     linewidth=LINE_W + 0.2,
                     label=label)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#DDD", linewidth=0.4, linestyle="--", zorder=0)
        ax.set_title("(b) ROC curve", pad=4, loc="left")
        ax.legend(loc="lower right", frameon=True, framealpha=0.92,
                   edgecolor="#CCC", fontsize=LEGEND_SIZE - 1)

        # Caption-style annotation about score source
        if any(c[8] == "confidence" for c in curves):
            fig.text(0.5, -0.03,
                     "* Confidence-derived score: API does not expose token "
                     "logprobs (Claude Sonnet 4.6) or constrained-scoring pass "
                     "produced ambiguous logits for this model/prompt "
                     "combination (others).",
                     ha="center", fontsize=LEGEND_SIZE - 1, style="italic",
                     color="#555")

        fig.tight_layout(pad=0.4)
        save_fig(fig, fig_path)


def write_main_table(metrics, path: Path):
    rows = []
    for m in MODEL_ORDER:
        for cot_flag in [False, True]:
            for s in STRATEGIES:
                k = run_key(m, s, cot_flag)
                if k in metrics:
                    rows.append((m, s, cot_flag, metrics[k]))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Classification results across all evaluated models on the CTAF-KHAF benchmark (94 test scenarios). Best macro-F$_1$ per model is \textbf{bold}.}",
        r"\label{tab:main}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Strategy & Acc. & Macro-F$_1$ & Parse-OK & Lat (s) \\",
        r"\midrule",
    ]
    cur_model = None
    best_f1_by_model = {}
    for m, s, cot, mm in rows:
        best_f1_by_model[m] = max(best_f1_by_model.get(m, 0), mm["macro_f1"])
    for m, s, cot, mm in rows:
        if m != cur_model:
            if cur_model is not None:
                lines.append(r"\midrule")
            cur_model = m
        is_best = abs(mm["macro_f1"] - best_f1_by_model[m]) < 1e-9
        f1str = f"\\textbf{{{mm['macro_f1']:.3f}}}" if is_best else f"{mm['macro_f1']:.3f}"
        strat_str = STRAT_NAMES[s] + (" + CoT" if cot else "")
        lines.append(
            f"{MODEL_NAMES[m]} & {strat_str} & {mm['accuracy']:.3f} & "
            f"{f1str} & {mm['parse_ok']:.2f} & {mm['avg_latency_s']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved -> {path.name}")


def write_summary_csv(metrics, path: Path):
    base_fields = ["run_key", "model", "strategy", "cot",
                   "accuracy", "macro_p", "macro_r", "macro_f1",
                   "parse_ok", "avg_latency_s", "n"]
    f1_fields = [f"f1_{l}" for l in LABELS]
    fields = base_fields + f1_fields
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in sorted(metrics.keys()):
            mm = metrics[k]
            row = {
                "run_key": k,
                "model": mm["model_key"],
                "strategy": mm["strategy"],
                "cot": mm["cot"],
                "accuracy": f"{mm['accuracy']:.4f}",
                "macro_p": f"{mm['macro_p']:.4f}",
                "macro_r": f"{mm['macro_r']:.4f}",
                "macro_f1": f"{mm['macro_f1']:.4f}",
                "parse_ok": f"{mm['parse_ok']:.4f}",
                "avg_latency_s": f"{mm['avg_latency_s']:.3f}",
                "n": mm["n"],
            }
            for l in LABELS:
                row[f"f1_{l}"] = f"{mm['per_class'][l]['f1']:.4f}"
            w.writerow(row)
    print(f"  Saved -> {path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="results/raw")
    p.add_argument("--out", default="results")
    p.add_argument("--binary", action="store_true",
                   help="Force binary mode (auto-detected by default)")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)

    # Auto-detect binary mode from raw records.
    binary_mode = args.binary or detect_binary_mode(raw_dir)
    if binary_mode:
        global LABELS
        LABELS = LABELS_BINARY
        print(f"  Detected BINARY mode -> classes: {LABELS}")

    print(f"Rebuilding metrics from {raw_dir}/")
    metrics, per_hazard = rebuild_all_metrics(raw_dir)
    print(f"  -> {len(metrics)} runs computed")

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "all_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(metrics_dir / "per_hazard_metrics.json", "w") as f:
        json.dump(per_hazard, f, indent=2)
    print(f"  Saved -> {metrics_dir}/all_metrics.json")
    print(f"  Saved -> {metrics_dir}/per_hazard_metrics.json")

    write_summary_csv(metrics, metrics_dir / "results_summary.csv")

    fig_dir = out_dir / "figures"
    print(f"\nGenerating figures in {fig_dir}/")
    fig_macro_f1_panels(metrics, fig_dir)
    fig_strategy_lines(metrics, fig_dir)
    fig_perclass_heatmap(metrics, fig_dir)
    fig_per_hazard(per_hazard, metrics, fig_dir)
    fig_latency(metrics, fig_dir)
    fig_cot_delta(metrics, fig_dir)
    fig_confusion_matrices(metrics, fig_dir)
    if binary_mode:
        fig_pr_roc(raw_dir, metrics, fig_dir, positive_class="danger")

    tab_dir = out_dir / "tables"
    print(f"\nGenerating LaTeX tables in {tab_dir}/")
    write_main_table(metrics, tab_dir / "table1_main_results.tex")

    print("\nDone.")
    print(f"  Total runs:    {len(metrics)}")
    print(f"  Models found:  {sorted({m['model_key'] for m in metrics.values()})}")
    print(f"  CoT runs:      {sum(1 for m in metrics.values() if m['cot'])}")
    print(f"  Direct runs:   {sum(1 for m in metrics.values() if not m['cot'])}")


if __name__ == "__main__":
    main()
