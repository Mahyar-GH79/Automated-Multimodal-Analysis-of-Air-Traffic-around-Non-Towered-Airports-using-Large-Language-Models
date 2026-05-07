#!/usr/bin/env python3
"""Ablation 3 — Text Masking (Missing Communication)"""

import os, sys, json, re, time, copy, random, argparse, logging, csv
from pathlib import Path
from datetime import datetime

import numpy as np

# OpenAI dispatch
try:
    from openai_inference import (
        is_openai_model, load_openai_client, run_openai_inference,
    )
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    def is_openai_model(k): return False


MODELS = {
    "qwen":    {"hf_id":"Qwen/Qwen2.5-7B-Instruct",
                "short_name":"Qwen2.5-7B","display":"Qwen 2.5-7B-Instruct",
                "max_new_tokens":120,"temperature":0.0},
    "mistral": {"hf_id":"mistralai/Mistral-7B-Instruct-v0.3",
                "short_name":"Mistral-7B","display":"Mistral-7B-Instruct-v0.3",
                "max_new_tokens":120,"temperature":0.0},
    "gemma":   {"hf_id":"google/gemma-2-9b-it",
                "short_name":"Gemma-2-9B","display":"Gemma-2-9B-IT",
                "max_new_tokens":120,"temperature":0.0,"use_cache":False},
    "gpt-4o":  {"openai_id":"gpt-4o","short_name":"GPT-4o","display":"GPT-4o",
                "max_new_tokens":120,"temperature":0.0,"is_openai":True},
    "gpt-5.4": {"openai_id":"gpt-5.4","short_name":"GPT-5.4","display":"GPT-5.4",
                "max_new_tokens":120,"temperature":0.0,"is_openai":True},
}
LABELS       = ["nominal","warning","hazard"]
MASK_RATES   = [10, 20, 40, 60, 80]     # percent
MASK_TOKEN   = "[MASKED]"
MASK_TYPES   = ["word", "utterance"]     # two masking strategies

ICL_EXAMPLE_IDS = {
    "nominal_1":"S074","nominal_2":"S096",
    "warning_1":"S050","warning_2":"S036",
    "hazard_1": "S031","hazard_2": "S003",
}

SYSTEM_PROMPT = """You are an automated aviation safety monitoring system for Half Moon Bay Airport (KHAF), a non-towered airport near San Francisco, California.

Classify the safety level of the CTAF radio communication as exactly ONE of:
  nominal  — Normal operations. No conflicts.
  warning  — Potential conflict or communication gap. Recoverable.
  hazard   — Imminent safety risk. Immediate action required.

Note: Some portions of the transcript may be missing or marked as [MASKED]
due to radio communication dropouts. Reason carefully from available context.

Respond with ONLY this JSON structure:
{
  "label": "<nominal|warning|hazard>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence>"
}"""


def setup_logging(out_dir):
    log = logging.getLogger("mask_abl")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%H:%M:%S")
    log.addHandler(logging.StreamHandler())
    fh = logging.FileHandler(out_dir / "ablation_mask.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    for h in log.handlers:
        h.setFormatter(fmt)
    return log


def mask_words(transcript_entries: list, rate: float, rng: random.Random) -> str:
    """Replace `rate` fraction of words across all entries with MASK_TOKEN."""
    lines = []
    for e in transcript_entries:
        words    = e["text"].split()
        n_mask   = max(0, round(len(words) * rate))
        mask_idx = set(rng.sample(range(len(words)), min(n_mask, len(words))))
        masked   = [MASK_TOKEN if i in mask_idx else w
                    for i, w in enumerate(words)]
        ts       = e.get("timestamp", "")
        lines.append(f"[{ts}] {' '.join(masked)}")
    return "\n".join(lines)


def mask_utterances(transcript_entries: list, rate: float,
                    rng: random.Random) -> str:
    """Drop `rate` fraction of entire transcript entries (missed transmissions)."""
    n        = len(transcript_entries)
    n_drop   = max(0, round(n * rate))
    drop_idx = set(rng.sample(range(n), min(n_drop, n)))
    lines    = []
    for i, e in enumerate(transcript_entries):
        ts = e.get("timestamp","")
        if i in drop_idx:
            lines.append(f"[{ts}] [MASKED TRANSMISSION]")
        else:
            lines.append(f"[{ts}] {e['text']}")
    return "\n".join(lines)


def apply_masking(scenario: dict, mask_type: str, rate: float,
                  rng: random.Random) -> str:
    """Return masked transcript text string."""
    entries = scenario["transcript_ground_truth"]
    rate_f  = rate / 100.0
    if mask_type == "word":
        return mask_words(entries, rate_f, rng)
    elif mask_type == "utterance":
        return mask_utterances(entries, rate_f, rng)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")


def format_input(scenario, masked_text, mask_type, rate):
    metar = scenario["metar"]
    mb = (f"METAR: {metar['raw']}\n"
          f"Conditions: {metar['description']} | Wind: {metar['wind_kt']} kt")
    desc = ("word-level" if mask_type == "word" else "utterance-level")
    return (f"--- WEATHER ---\n{mb}\n\n"
            f"--- CTAF TRANSCRIPT ({desc} masking, {rate}% missing) ---\n"
            f"{masked_text}\n\n"
            f"Classify the safety level:")

def build_zero_shot(scenario, masked_text, model_key, mask_type, rate):
    NO_SYS  = {"gemma"}
    content = format_input(scenario, masked_text, mask_type, rate)
    if model_key not in NO_SYS:
        return [{"role":"system","content":SYSTEM_PROMPT},
                {"role":"user",  "content":content}]
    return [{"role":"user","content":f"{SYSTEM_PROMPT}\n\n{content}"}]

_SYNONYMS = {
    "safe":"nominal","normal":"nominal","clear":"nominal",
    "caution":"warning","warn":"warning","alert":"warning",
    "danger":"hazard","critical":"hazard","emergency":"hazard","unsafe":"hazard",
}
def parse_response(raw):
    clean = re.sub(r"```[a-z]*\n?","",raw).strip()
    try:
        m = re.search(r"\{.*?\}", clean, re.DOTALL)
        if m:
            obj   = json.loads(m.group(0))
            label = str(obj.get("label","")).strip().lower()
            label = _SYNONYMS.get(label,label)
            if label not in LABELS:
                for lbl in ["hazard","warning","nominal"]:
                    if lbl in clean.lower(): label=lbl; break
            return {"label":label,"confidence":float(obj.get("confidence",0.5)),
                    "reasoning":str(obj.get("reasoning",""))[:200],
                    "parse_ok":label in LABELS}
    except Exception:
        pass
    for lbl in ["hazard","warning","nominal"]:
        if lbl in clean.lower():
            return {"label":lbl,"confidence":0.5,
                    "reasoning":clean[:200],"parse_ok":True}
    return {"label":"unknown","confidence":0.0,
            "reasoning":clean[:200],"parse_ok":False}

def load_model(model_key, log):
    cfg = MODELS[model_key]
    if cfg.get("is_openai"):
        log.info(f"Initializing OpenAI client for {cfg['display']}")
        return load_openai_client(), None

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log.info(f"Loading {cfg['display']}")
    tok = AutoTokenizer.from_pretrained(
        cfg["hf_id"], trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(trust_remote_code=True, device_map="auto",
              dtype=__import__("torch").float16,
              token=os.environ.get("HF_TOKEN"))
    if model_key == "gemma":
        kw["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], **kw)
    model.eval()
    log.info(f"  Loaded on {next(model.parameters()).device}")
    return model, tok

def run_inference(model, tok, messages, cfg):
    if cfg.get("is_openai"):
        return run_openai_inference(model, messages, cfg["openai_id"])

    import torch
    t0 = time.time()
    chat_out = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt")
    if hasattr(chat_out,"input_ids"):
        ids  = chat_out.input_ids.to(model.device)
        mask = getattr(chat_out,"attention_mask",None)
        if mask is not None: mask = mask.to(model.device)
    elif isinstance(chat_out,dict):
        ids  = chat_out["input_ids"].to(model.device)
        mask = chat_out.get("attention_mask")
        if mask is not None: mask = mask.to(model.device)
    else:
        ids  = chat_out.to(model.device); mask = None
    prompt_len = ids.shape[-1]
    gkw = dict(input_ids=ids, max_new_tokens=cfg["max_new_tokens"],
               do_sample=False, pad_token_id=tok.pad_token_id,
               eos_token_id=tok.eos_token_id)
    if mask is not None: gkw["attention_mask"] = mask
    if cfg.get("use_cache") is False: gkw["use_cache"] = False
    with torch.no_grad():
        out = model.generate(**gkw)
    raw = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    return raw, time.time() - t0

def compute_metrics(y_true, y_pred):
    cm  = np.zeros((3,3),dtype=int)
    l2i = {l:i for i,l in enumerate(LABELS)}
    for yt,yp in zip(y_true,y_pred):
        ti,pi = l2i.get(yt,-1),l2i.get(yp,-1)
        if ti>=0 and pi>=0: cm[ti,pi]+=1
    pc = {}
    for i,lbl in enumerate(LABELS):
        tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp
        p=tp/(tp+fp) if (tp+fp) else 0.
        r=tp/(tp+fn) if (tp+fn) else 0.
        f=2*p*r/(p+r) if (p+r) else 0.
        pc[lbl]={"precision":round(p,4),"recall":round(r,4),
                 "f1":round(f,4),"support":int(cm[i,:].sum())}
    n=len(y_true)
    mf=np.mean([pc[l]["f1"] for l in LABELS])
    ac=cm.diagonal().sum()/n if n else 0.
    return {"accuracy":round(ac,4),"macro_f1":round(mf,4),
            "per_class":pc,"confusion_matrix":cm.tolist(),"n":n}


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl

MODEL_COLORS = {"qwen":"#2166AC","mistral":"#762A83","gemma":"#1B7837",
                "gpt-4o":"#B35806","gpt-5.4":"#000000"}
MODEL_MARKERS = {"qwen":"o","mistral":"s","gemma":"^","gpt-4o":"D","gpt-5.4":"*"}
MODEL_MARKERS= {"qwen":"o","mistral":"s","gemma":"^"}
MODEL_NAMES  = {"qwen":"Qwen 2.5-7B","mistral":"Mistral-7B","gemma":"Gemma-2-9B",
                "gpt-4o":"GPT-4o","gpt-5.4":"GPT-5.4"}
TYPE_STYLES  = {"word":"-","utterance":"--"}
TYPE_LABELS  = {"word":"Word masking","utterance":"Utterance masking"}

ICML_RC = {
    "font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
    "font.size":8.5,"axes.titlesize":9.5,"axes.labelsize":8.5,
    "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.5,
    "lines.linewidth":1.2,"lines.markersize":5,
    "axes.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":False,
    "xtick.major.width":0.6,"ytick.major.width":0.6,
    "xtick.major.size":2.5,"ytick.major.size":2.5,
    "xtick.minor.size":0,"ytick.minor.size":0,
    "xtick.direction":"out","ytick.direction":"out",
    "figure.dpi":300,"savefig.dpi":300,
    "savefig.bbox":"tight","savefig.pad_inches":0.02,
}

def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path.name}")


def plot_mask_f1_curve(all_metrics, mask_rates, gt_metrics, out_path):
    """Two panels: word masking (left) and utterance masking (right)."""
    with mpl.rc_context(ICML_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)

        for pi, mtype in enumerate(MASK_TYPES):
            ax = axes[pi]
            for mk in ["qwen","mistral","gemma"]:
                ys = [all_metrics.get(f"{mk}_zero_shot_mask_{mtype}_{r}",{})
                                 .get("macro_f1", float("nan"))
                      for r in mask_rates]
                ax.plot(mask_rates, ys,
                        color=MODEL_COLORS[mk],
                        marker=MODEL_MARKERS[mk],
                        markersize=5, linewidth=1.2,
                        label=MODEL_NAMES[mk], zorder=4)
                for xi, yi in zip(mask_rates, ys):
                    if not np.isnan(yi):
                        ax.annotate(f"{yi:.2f}", (xi, yi),
                                    xytext=(0,5), textcoords="offset points",
                                    ha="center", fontsize=5.5,
                                    color=MODEL_COLORS[mk])

                # GT baseline (dotted)
                gt_f1 = gt_metrics.get(f"{mk}_zero_shot",{}).get("macro_f1")
                if gt_f1:
                    ax.axhline(gt_f1, color=MODEL_COLORS[mk],
                               linewidth=0.7, linestyle=":", alpha=0.55)

            ax.set_xticks(mask_rates)
            ax.set_xticklabels([f"{r}\\%" for r in mask_rates])
            ax.set_xlabel("Masking rate")
            ax.set_ylim(0.0, 0.85)
            ax.grid(axis="y", color="#CCCCCC", linewidth=0.4, linestyle="--")
            ax.set_title(TYPE_LABELS[mtype], fontsize=8.5, pad=3)
            if pi == 0:
                ax.set_ylabel("Macro-averaged F$_1$")
                ax.legend(loc="upper right", frameon=True,
                          framealpha=0.9, edgecolor="#CCCCCC")
            ax.text(0.98, 0.97, "Dotted = GT baseline",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=5.5, color="#888888", style="italic")

        fig.suptitle("Text masking ablation: macro-F$_1$ vs.~masking rate (zero-shot)",
                     fontsize=9.5, y=1.02, x=0.02, ha="left")
        save_fig(fig, out_path)


def plot_mask_perclass(all_metrics, mask_rates, out_path):
    """3×2 grid: rows = safety class, cols = masking type."""
    with mpl.rc_context(ICML_RC):
        fig, axes = plt.subplots(3, 2, figsize=(6.0, 5.5),
                                 sharey="row", sharex=True)

        for li, lbl in enumerate(LABELS):
            for pi, mtype in enumerate(MASK_TYPES):
                ax = axes[li][pi]
                for mk in ["qwen","mistral","gemma"]:
                    ys = [all_metrics.get(
                              f"{mk}_zero_shot_mask_{mtype}_{r}",{})
                              .get("per_class",{}).get(lbl,{})
                              .get("f1", float("nan"))
                          for r in mask_rates]
                    ax.plot(mask_rates, ys,
                            color=MODEL_COLORS[mk],
                            marker=MODEL_MARKERS[mk],
                            markersize=3.5, linewidth=1.0,
                            label=MODEL_NAMES[mk])

                ax.set_ylim(0, 1.05)
                ax.grid(axis="y", color="#CCCCCC", linewidth=0.3,
                        linestyle="--")
                if li == 0:
                    ax.set_title(TYPE_LABELS[mtype], fontsize=7.5, pad=2)
                if li == 2:
                    ax.set_xticks(mask_rates)
                    ax.set_xticklabels([f"{r}\\%" for r in mask_rates],
                                       fontsize=5.5)
                    ax.set_xlabel("Masking rate", fontsize=7.0)
                if pi == 0:
                    ax.set_ylabel(lbl.capitalize() + "\nF$_1$",
                                  fontsize=7.0)

        # Shared legend
        handles = [plt.Line2D([0],[0], color=MODEL_COLORS[mk],
                               marker=MODEL_MARKERS[mk], markersize=4,
                               label=MODEL_NAMES[mk])
                   for mk in ["qwen","mistral","gemma"]]
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
                   bbox_to_anchor=(0.5, -0.03), fontsize=7.0)

        fig.suptitle("Per-class F$_1$ under text masking (zero-shot)",
                     fontsize=9.5, y=1.01)
        save_fig(fig, out_path)


def plot_mask_heatmap(all_metrics, mask_rates, out_path):
    """Heatmap: rows = model, cols = masking rate, value = macro F1."""
    models_list = ["qwen","mistral","gemma"]

    with mpl.rc_context(ICML_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))

        for pi, mtype in enumerate(MASK_TYPES):
            ax  = axes[pi]
            mat = np.array([
                [all_metrics.get(f"{mk}_zero_shot_mask_{mtype}_{r}",{})
                            .get("macro_f1", 0)
                 for r in mask_rates]
                for mk in models_list
            ])  # (3 models, 5 rates)

            im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=0.85,
                           aspect="auto")

            ax.set_xticks(range(len(mask_rates)))
            ax.set_xticklabels([f"{r}\\%" for r in mask_rates], fontsize=6.5)
            ax.set_yticks(range(len(models_list)))
            ax.set_yticklabels([MODEL_NAMES[mk] for mk in models_list],
                               fontsize=6.5)
            ax.set_title(TYPE_LABELS[mtype], fontsize=8.0, pad=3)
            ax.grid(False)

            for i in range(len(models_list)):
                for j in range(len(mask_rates)):
                    v  = mat[i,j]
                    tc = "white" if (v < 0.25 or v > 0.70) else "#1a1a1a"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5, color=tc)

        cbar = plt.colorbar(im, ax=axes.ravel().tolist(),
                            fraction=0.02, pad=0.02)
        cbar.ax.tick_params(labelsize=6.5)
        cbar.set_label("Macro F$_1$", fontsize=7.0)

        fig.suptitle("Macro-F$_1$ heatmap: text masking ablation (zero-shot)",
                     fontsize=9.5, y=1.03, x=0.02, ha="left")
        save_fig(fig, out_path)


def make_mask_table(all_metrics, mask_rates, gt_metrics):
    rate_heads = " & ".join(f"{r}\\%" for r in mask_rates)
    lines = [
        r"\begin{table*}[t]",r"\centering",
        r"\caption{Text masking ablation: macro-F$_1$ at increasing masking rates "
        r"for word-level and utterance-level masking. "
        r"GT = unmasked ground-truth text. Zero-shot prompting.}",
        r"\label{tab:mask_ablation}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lll c" + "c"*len(mask_rates) + r"}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Mask type} & \textbf{GT} & "
        + rate_heads + r" \\",
        r"\midrule",
    ]
    for mk in ["qwen","mistral","gemma"]:
        gt_f1 = gt_metrics.get(f"{mk}_zero_shot",{}).get("macro_f1",0)
        for ti, mtype in enumerate(MASK_TYPES):
            vals = [all_metrics.get(f"{mk}_zero_shot_mask_{mtype}_{r}",{})
                               .get("macro_f1",0) for r in mask_rates]
            all_v = [gt_f1] + vals
            best  = max(all_v)
            def b(v):
                s = f"{v:.3f}"
                return r"\textbf{" + s + r"}" if abs(v-best)<1e-6 else s
            mc  = (r"\multirow{2}{*}{" + MODEL_NAMES[mk] + r"}"
                   if ti == 0 else "")
            nl  = r"\\"
            lines.append(
                f"  {mc} & {TYPE_LABELS[mtype]} & {b(gt_f1)} & "
                + " & ".join(b(v) for v in vals) + f" {nl}")
        lines.append(r"  \midrule")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    required=True)
    ap.add_argument("--gt-metrics", required=True)
    ap.add_argument("--out",        default="results/ablation_mask")
    ap.add_argument("--mask-rates", nargs="+", type=int, default=MASK_RATES)
    ap.add_argument("--mask-types", nargs="+", default=MASK_TYPES,
                    choices=MASK_TYPES)
    ap.add_argument("--models",     nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out)

    log.info("=" * 60)
    log.info("Ablation 3 — Text Masking")
    log.info(f"  Mask rates : {args.mask_rates}%")
    log.info(f"  Mask types : {args.mask_types}")
    log.info(f"  Models     : {args.models}")
    log.info("=" * 60)

    with open(args.dataset) as f:
        data = json.load(f)
    all_scenarios = data["scenarios"]
    icl_ids  = set(ICL_EXAMPLE_IDS.values())
    test_set = [s for s in all_scenarios if s["scenario_id"] not in icl_ids]

    with open(args.gt_metrics) as f:
        gt_metrics = json.load(f)

    all_metrics = {}

    if args.plots_only:
        met_path = out / "metrics" / "mask_all_metrics.json"
        if met_path.exists():
            with open(met_path) as f:
                all_metrics = json.load(f)
        _save_outputs(all_metrics, args.mask_rates, gt_metrics, out, log)
        return

    # Pre-generate all masked texts with fixed RNG (reproducible)
    # masked_texts[mtype][rate][sid] = text
    masked_texts = {mt: {r: {} for r in args.mask_rates}
                    for mt in args.mask_types}
    for mtype in args.mask_types:
        for rate in args.mask_rates:
            for sc in all_scenarios:
                sid = sc["scenario_id"]
                # Use deterministic seed per (scenario, type, rate)
                sc_rng = random.Random(args.seed + hash(sid) + rate)
                masked_texts[mtype][rate][sid] = apply_masking(
                    sc, mtype, rate, sc_rng)
    log.info("All masked transcripts generated (deterministic)")

    import torch
    for model_key in args.models:
        log.info(f"\n{'='*60}\nModel: {MODELS[model_key]['display']}")
        model, tok = load_model(model_key, log)

        for mtype in args.mask_types:
            for rate in args.mask_rates:
                cond_tag = f"mask_{mtype}_{rate}"
                run_key  = f"{model_key}_zero_shot_{cond_tag}"
                raw_path = out / "raw" / f"{run_key}.json"
                raw_path.parent.mkdir(parents=True, exist_ok=True)

                if raw_path.exists():
                    log.info(f"  {run_key} [RESUMING]")
                    with open(raw_path) as f:
                        saved = json.load(f)
                    records   = saved["records"]
                    y_true    = [r["ground_truth"] for r in records]
                    y_pred    = [r["predicted"]    for r in records]
                    latencies = [r["latency_s"]    for r in records]
                    m         = compute_metrics(y_true, y_pred)
                    m["avg_latency_s"] = round(float(np.mean(latencies)),3)
                    m.update({"model":MODELS[model_key]["display"],
                               "model_key":model_key,"strategy":"zero_shot",
                               "condition":cond_tag,"run_key":run_key,
                               "mask_type":mtype,"mask_rate_pct":rate})
                    all_metrics[run_key] = m
                    continue

                log.info(f"  {mtype} masking {rate}%  ({len(test_set)} scenarios)")
                records, y_true, y_pred, latencies = [], [], [], []

                for idx, sc in enumerate(test_set):
                    sid    = sc["scenario_id"]
                    masked = masked_texts[mtype][rate][sid]
                    msgs   = build_zero_shot(sc, masked, model_key, mtype, rate)
                    raw_out, lat = run_inference(model, tok, msgs,
                                                MODELS[model_key])
                    parsed = parse_response(raw_out)
                    gt     = sc["label"]

                    # Compute actual masking stats
                    orig_words   = sum(len(e["text"].split())
                                      for e in sc["transcript_ground_truth"])
                    masked_count = masked.count(MASK_TOKEN) + \
                                   masked.count("MASKED TRANSMISSION")

                    rec = {
                        "scenario_id":   sid,
                        "hazard_type":   sc["hazard_type"],
                        "ground_truth":  gt,
                        "predicted":     parsed["label"],
                        "confidence":    parsed["confidence"],
                        "reasoning":     parsed["reasoning"],
                        "raw_output":    raw_out,
                        "latency_s":     round(lat, 3),
                        "correct":       gt == parsed["label"],
                        "model":         MODELS[model_key]["short_name"],
                        "mask_type":     mtype,
                        "mask_rate_pct": rate,
                        "orig_words":    orig_words,
                        "masked_tokens": masked_count,
                        "parse_ok":      parsed["parse_ok"],
                    }
                    records.append(rec)
                    y_true.append(gt); y_pred.append(parsed["label"])
                    latencies.append(lat)
                    status = "✓" if rec["correct"] else "✗"
                    log.info(f"    [{idx+1:3d}/{len(test_set)}] {sid} "
                             f"GT={gt:7s} PRED={parsed['label']:7s} {status} "
                             f"({lat:.2f}s, {masked_count} masked)")

                with open(raw_path, "w") as f:
                    json.dump({"run_key":run_key,"condition":cond_tag,
                               "mask_type":mtype,"mask_rate_pct":rate,
                               "model":MODELS[model_key]["display"],
                               "timestamp":datetime.now().isoformat(),
                               "records":records}, f, indent=2)

                m = compute_metrics(y_true, y_pred)
                m["avg_latency_s"] = round(float(np.mean(latencies)),3)
                m.update({"model":MODELS[model_key]["display"],
                           "model_key":model_key,"strategy":"zero_shot",
                           "condition":cond_tag,"run_key":run_key,
                           "mask_type":mtype,"mask_rate_pct":rate})
                all_metrics[run_key] = m
                log.info(f"    → Acc={m['accuracy']:.3f}  "
                         f"MacroF1={m['macro_f1']:.3f}")

        if not MODELS[model_key].get("is_openai"):
            del model, tok
            torch.cuda.empty_cache()
            log.info(f"  GPU freed after {MODELS[model_key]['display']}")

    _save_outputs(all_metrics, args.mask_rates, gt_metrics, out, log)


def _save_outputs(all_metrics, mask_rates, gt_metrics, out, log):
    met_dir = out / "metrics"
    met_dir.mkdir(exist_ok=True)
    with open(met_dir/"mask_all_metrics.json","w") as f:
        json.dump(all_metrics, f, indent=2)

    rows = []
    for key, m in all_metrics.items():
        rows.append({"run_key":key,"model":m.get("model",""),
                     "mask_type":m.get("mask_type",""),
                     "mask_rate":m.get("mask_rate_pct",""),
                     "accuracy":m.get("accuracy",0),
                     "macro_f1":m.get("macro_f1",0)})
    if rows:
        with open(met_dir/"mask_summary.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_mask_f1_curve(all_metrics, mask_rates, gt_metrics,
                       fig_dir/"fig_mask_f1_curve.pdf")
    plot_mask_perclass(all_metrics, mask_rates,
                       fig_dir/"fig_mask_perclass.pdf")
    plot_mask_heatmap(all_metrics, mask_rates,
                      fig_dir/"fig_mask_heatmap.pdf")

    tab_dir = out / "tables"
    tab_dir.mkdir(exist_ok=True)
    (tab_dir/"table_mask_ablation.tex").write_text(
        make_mask_table(all_metrics, mask_rates, gt_metrics))

    log.info(f"\nSaved all masking ablation outputs to {out}/")
    log.info(f"  {'Run key':<50} {'F1':>6}")
    for key,m in sorted(all_metrics.items(),
                         key=lambda x:x[1].get("macro_f1",0),reverse=True)[:10]:
        log.info(f"  {key:<50} {m.get('macro_f1',0):>6.3f}")


if __name__ == "__main__":
    main()