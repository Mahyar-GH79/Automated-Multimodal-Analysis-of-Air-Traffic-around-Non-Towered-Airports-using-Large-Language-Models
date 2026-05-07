#!/usr/bin/env python3
"""Ablation 1 — Whisper ASR Quality"""

import os, sys, json, re, time, copy, random, argparse, logging, csv
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

# OpenAI dispatch (graceful if module/key missing)
try:
    from openai_inference import (
        is_openai_model, load_openai_client, run_openai_inference,
    )
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    def is_openai_model(k): return False


MODELS = {
    "qwen":    {"hf_id": "Qwen/Qwen2.5-7B-Instruct",
                "short_name": "Qwen2.5-7B", "display": "Qwen 2.5-7B-Instruct",
                "max_new_tokens": 120, "temperature": 0.0},
    "mistral": {"hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
                "short_name": "Mistral-7B", "display": "Mistral-7B-Instruct-v0.3",
                "max_new_tokens": 120, "temperature": 0.0},
    "gemma":   {"hf_id": "google/gemma-2-9b-it",
                "short_name": "Gemma-2-9B", "display": "Gemma-2-9B-IT",
                "max_new_tokens": 120, "temperature": 0.0,
                "use_cache": False},
    "gpt-4o":  {"openai_id": "gpt-4o",
                "short_name": "GPT-4o", "display": "GPT-4o",
                "max_new_tokens": 120, "temperature": 0.0, "is_openai": True},
    "gpt-5.4": {"openai_id": "gpt-5.4",
                "short_name": "GPT-5.4", "display": "GPT-5.4",
                "max_new_tokens": 120, "temperature": 0.0, "is_openai": True},
}
STRATEGIES  = ["zero_shot", "one_shot", "few_shot"]
LABELS      = ["nominal", "warning", "hazard"]
WHISPER_SIZES = ["base", "medium", "large-v3"]

ICL_EXAMPLE_IDS = {
    "nominal_1": "S074", "nominal_2": "S096",
    "warning_1": "S050", "warning_2": "S036",
    "hazard_1":  "S031", "hazard_2":  "S003",
}

SYSTEM_PROMPT = """You are an automated aviation safety monitoring system for Half Moon Bay Airport (KHAF), a non-towered airport near San Francisco, California.

Your task is to analyze CTAF (Common Traffic Advisory Frequency) radio communications at KHAF and classify the safety status of the current traffic situation.

You will be given:
1. METAR weather data for KHAF
2. A CTAF radio transcript (may contain ASR transcription artifacts)

Classify the safety level as exactly ONE of:
  nominal  — Normal operations. All required position calls are present. No traffic conflicts.
  warning  — Potential conflict or communication gap. Recoverable with pilot action.
  hazard   — Imminent safety risk. Immediate action required to prevent collision or incident.

CTAF communication rules at non-towered airports (FAA AC 90-66C):
- Pilots MUST self-announce on each pattern leg: crosswind, downwind, base, final.
- Straight-in approaches: announce at 10, 5, and 3 NM.
- Runway clear announcement required after landing.
- Go-around MUST be announced immediately.

Respond with ONLY this JSON structure (no other text):
{
  "label": "<nominal|warning|hazard>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence explaining the key safety factor>"
}"""

NATO = {
    "alpha":"A","bravo":"B","charlie":"C","delta":"D","echo":"E",
    "foxtrot":"F","golf":"G","hotel":"H","india":"I","juliet":"J",
    "kilo":"K","lima":"L","mike":"M","november":"N","oscar":"O",
    "papa":"P","quebec":"Q","romeo":"R","sierra":"S","tango":"T",
    "uniform":"U","victor":"V","whiskey":"W","xray":"X",
    "yankee":"Y","zulu":"Z",
}
_NATO_PAT = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in NATO) + r")\b", re.IGNORECASE)

def translate_phonetic(text):
    return _NATO_PAT.sub(lambda m: NATO[m.group(0).lower()], text)


def setup_logging(out_dir):
    log = logging.getLogger("whisper_abl")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%H:%M:%S")
    log.addHandler(logging.StreamHandler())
    log.addHandler(logging.FileHandler(out_dir / "ablation_whisper.log"))
    for h in log.handlers:
        h.setFormatter(fmt)
    return log


def merge_segments(segs, max_len=80, max_gap=2.0):
    if not segs:
        return []
    merged, cur = [], None
    for s in segs:
        txt = s["text"].strip()
        if not txt:
            continue
        if cur is None:
            cur = {"text": txt, "start": s["start"], "end": s["end"]}
            continue
        if s["start"] - cur["end"] < max_gap and len(cur["text"] + " " + txt) < max_len:
            cur["text"] = (cur["text"] + " " + txt).strip()
            cur["end"]  = s["end"]
        else:
            merged.append(copy.deepcopy(cur))
            cur = {"text": txt, "start": s["start"], "end": s["end"]}
    if cur:
        merged.append(cur)
    return merged


def transcribe_audio(audio_path: Path, whisper_size: str, log) -> str:
    """Return plain-text transcript (NATO-translated) from audio file."""
    from faster_whisper import WhisperModel
    import torch

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(whisper_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"threshold": 0.6, "min_speech_duration_ms": 250,
                         "max_speech_duration_s": float("inf"),
                         "min_silence_duration_ms": 100, "speech_pad_ms": 400},
    )
    word_segs = []
    for seg in segments:
        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                if w.word.strip():
                    word_segs.append({"text": w.word.strip(),
                                      "start": w.start, "end": w.end})

    merged = merge_segments(word_segs)

    # Build SRT-format text
    try:
        import srt as srt_lib
        srt_list = [
            srt_lib.Subtitle(
                index=i,
                start=timedelta(seconds=v["start"]),
                end=timedelta(seconds=v["end"]),
                content=v["text"].strip(),
            )
            for i, v in enumerate(merged)
        ]
        raw_srt = srt_lib.compose(srt_list)
    except ImportError:
        raw_srt = "\n".join(v["text"] for v in merged)

    del model
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()

    return translate_phonetic(raw_srt)


def run_transcription_batch(scenarios, audio_dir, whisper_size, tx_dir, log):
    """Transcribe all scenarios with one Whisper size."""
    results = {}
    audio_dir = Path(audio_dir)
    tx_dir    = Path(tx_dir)
    tx_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Whisper-{whisper_size}: transcribing {len(scenarios)} scenarios")

    for i, sc in enumerate(scenarios):
        sid        = sc["scenario_id"]
        cache_path = tx_dir / f"{sid}_{whisper_size}.txt"

        if cache_path.exists():
            results[sid] = cache_path.read_text(encoding="utf-8")
            log.info(f"  [{i+1:3d}/{len(scenarios)}] {sid} whisper-{whisper_size} [cached]")
            continue

        audio_path = audio_dir / sid / "audio.mp3"
        if not audio_path.exists():
            log.warning(f"  [{i+1:3d}] {sid}: audio not found at {audio_path}")
            results[sid] = ""
            continue

        try:
            text = transcribe_audio(audio_path, whisper_size, log)
            cache_path.write_text(text, encoding="utf-8")
            results[sid] = text
            # Count words as quality proxy
            n_words = len(text.split())
            log.info(f"  [{i+1:3d}/{len(scenarios)}] {sid} whisper-{whisper_size} "
                     f"→ {n_words} words")
        except Exception as e:
            log.error(f"  [{i+1:3d}] {sid} whisper-{whisper_size} FAILED: {e}")
            results[sid] = ""

    return results


def format_scenario_input(scenario, transcript_override=None):
    metar = scenario["metar"]
    metar_block = f"METAR: {metar['raw']}\nConditions: {metar['description']}"
    if metar.get("ceiling_ft"):
        metar_block += f" | Ceiling: {metar['ceiling_ft']} ft"
    metar_block += f" | Wind: {metar['wind_kt']} kt"

    if transcript_override is not None:
        tx_lines = transcript_override
    else:
        tx_lines = "\n".join(
            f"[{e['timestamp']}] {e['text']}"
            for e in scenario["transcript_ground_truth"]
        )

    return (f"--- WEATHER ---\n{metar_block}\n\n"
            f"--- CTAF TRANSCRIPT ---\n{tx_lines}\n\n"
            f"Classify the safety level:")


def format_icl_example(scenario):
    user = format_scenario_input(scenario)
    asst = json.dumps({
        "label":      scenario["label"],
        "confidence": 0.95,
        "reasoning":  scenario["ground_truth_advisory"][:120].rstrip(".") + ".",
    })
    return user, asst


def build_messages(scenario, icl_ordered, strategy, model_key,
                   transcript_override=None):
    NO_SYSTEM = {"gemma"}
    asst_role = "model" if model_key == "gemma" else "assistant"
    use_sys   = model_key not in NO_SYSTEM

    msgs = []
    if use_sys:
        msgs.append({"role": "system", "content": SYSTEM_PROMPT})

    if strategy in ("one_shot", "few_shot"):
        n = 1 if strategy == "one_shot" else 2
        for label in LABELS:
            exs = [e for e in icl_ordered if e["label"] == label][:n]
            for ex in exs:
                u, a = format_icl_example(ex)
                msgs.append({"role": "user",      "content": u})
                msgs.append({"role": asst_role,   "content": a})

    sc_content = format_scenario_input(scenario, transcript_override)

    if use_sys:
        msgs.append({"role": "user", "content": sc_content})
    else:
        if msgs:
            first_idx = next((i for i, m in enumerate(msgs)
                              if m["role"] == "user"), None)
            if first_idx is not None:
                msgs[first_idx]["content"] = (
                    f"{SYSTEM_PROMPT}\n\n{msgs[first_idx]['content']}")
                msgs.append({"role": "user", "content": sc_content})
            else:
                msgs.append({"role": "user",
                             "content": f"{SYSTEM_PROMPT}\n\n{sc_content}"})
        else:
            msgs.append({"role": "user",
                         "content": f"{SYSTEM_PROMPT}\n\n{sc_content}"})
    return msgs


_SYNONYMS = {
    "safe":"nominal","normal":"nominal","clear":"nominal","ok":"nominal",
    "caution":"warning","warn":"warning","alert":"warning",
    "danger":"hazard","critical":"hazard","emergency":"hazard","unsafe":"hazard",
}

def parse_response(raw):
    clean = re.sub(r"```[a-z]*\n?", "", raw).strip()
    try:
        m = re.search(r"\{.*?\}", clean, re.DOTALL)
        if m:
            obj   = json.loads(m.group(0))
            label = str(obj.get("label","")).strip().lower()
            label = _SYNONYMS.get(label, label)
            if label not in LABELS:
                label = _extract_label(clean)
            return {"label": label,
                    "confidence": float(obj.get("confidence", 0.5)),
                    "reasoning":  str(obj.get("reasoning",""))[:200],
                    "parse_ok":   label in LABELS}
    except Exception:
        pass
    label = _extract_label(clean)
    return {"label": label, "confidence": 0.5,
            "reasoning": clean[:200], "parse_ok": label in LABELS}

def _extract_label(text):
    tl = text.lower()
    for lbl in ["hazard","warning","nominal"]:
        if lbl in tl:
            return lbl
    for syn, lbl in _SYNONYMS.items():
        if syn in tl:
            return lbl
    return "unknown"


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

    kwargs = dict(trust_remote_code=True, device_map="auto",
                  dtype=torch.float16,
                  token=os.environ.get("HF_TOKEN"))
    if model_key == "gemma":
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], **kwargs)
    model.eval()
    log.info(f"  Loaded on {next(model.parameters()).device}")
    return model, tok


def run_inference(model, tok, messages, cfg):
    if cfg.get("is_openai"):
        # `model` here is the OpenAI client; `tok` is None.
        return run_openai_inference(model, messages, cfg["openai_id"])

    import torch
    t0 = time.time()
    chat_out = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if hasattr(chat_out, "input_ids"):
        ids  = chat_out.input_ids.to(model.device)
        mask = getattr(chat_out, "attention_mask", None)
        if mask is not None:
            mask = mask.to(model.device)
    elif isinstance(chat_out, dict):
        ids  = chat_out["input_ids"].to(model.device)
        mask = chat_out.get("attention_mask")
        if mask is not None:
            mask = mask.to(model.device)
    else:
        ids  = chat_out.to(model.device)
        mask = None

    prompt_len = ids.shape[-1]
    gen_kw = dict(input_ids=ids, max_new_tokens=cfg["max_new_tokens"],
                  do_sample=cfg["temperature"] > 0,
                  pad_token_id=tok.pad_token_id,
                  eos_token_id=tok.eos_token_id)
    if mask is not None:
        gen_kw["attention_mask"] = mask
    if cfg.get("use_cache") is False:
        gen_kw["use_cache"] = False

    with torch.no_grad():
        out = model.generate(**gen_kw)
    raw = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    return raw, time.time() - t0


def compute_metrics(y_true, y_pred):
    cm    = np.zeros((3,3), dtype=int)
    l2i   = {l:i for i,l in enumerate(LABELS)}
    for yt, yp in zip(y_true, y_pred):
        ti, pi = l2i.get(yt,-1), l2i.get(yp,-1)
        if ti >= 0 and pi >= 0:
            cm[ti,pi] += 1
    pc = {}
    for i,lbl in enumerate(LABELS):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        p  = tp/(tp+fp) if (tp+fp) else 0.
        r  = tp/(tp+fn) if (tp+fn) else 0.
        f  = 2*p*r/(p+r) if (p+r) else 0.
        pc[lbl] = {"precision":round(p,4),"recall":round(r,4),
                   "f1":round(f,4),"support":int(cm[i,:].sum())}
    n       = len(y_true)
    macro_f = np.mean([pc[l]["f1"] for l in LABELS])
    acc     = cm.diagonal().sum()/n if n else 0.
    return {"accuracy":round(acc,4), "macro_f1":round(macro_f,4),
            "per_class":pc, "confusion_matrix":cm.tolist(), "n":n,
            "macro_p": round(np.mean([pc[l]["precision"] for l in LABELS]),4),
            "macro_r": round(np.mean([pc[l]["recall"]    for l in LABELS]),4)}


def evaluate_on_transcripts(
        model_key, model, tok, test_set, icl_ordered,
        transcripts_by_sid,   # dict sid->text, or None to use ground truth
        condition_tag,        # e.g. "whisper_base" or "ground_truth"
        out_dir, log):
    """
    Run all 3 strategies for one model on the given transcript source.
    Returns dict: strategy -> metrics
    """
    results = {}
    for strategy in STRATEGIES:
        run_key   = f"{model_key}_{strategy}_{condition_tag}"
        raw_path  = out_dir / "raw" / f"{run_key}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume
        if raw_path.exists():
            log.info(f"  [{run_key}] RESUMING from cache")
            with open(raw_path) as f:
                saved = json.load(f)
            records  = saved["records"]
            y_true   = [r["ground_truth"] for r in records]
            y_pred   = [r["predicted"]    for r in records]
            latencies= [r["latency_s"]    for r in records]
            m        = compute_metrics(y_true, y_pred)
            m["avg_latency_s"] = round(float(np.mean(latencies)),3)
            m.update({"model": MODELS[model_key]["display"],
                      "model_key": model_key, "strategy": strategy,
                      "condition": condition_tag, "run_key": run_key})
            results[strategy] = m
            continue

        log.info(f"  [{run_key}] running {len(test_set)} scenarios")
        records, y_true, y_pred, latencies = [], [], [], []

        for idx, sc in enumerate(test_set):
            sid  = sc["scenario_id"]
            tx   = transcripts_by_sid.get(sid) if transcripts_by_sid else None
            msgs = build_messages(sc, icl_ordered, strategy, model_key,
                                  transcript_override=tx)
            raw_out, lat = run_inference(model, tok, msgs, MODELS[model_key])
            parsed = parse_response(raw_out)
            gt     = sc["label"]
            rec    = {
                "scenario_id": sid, "hazard_type": sc["hazard_type"],
                "ground_truth": gt, "predicted": parsed["label"],
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"],
                "raw_output": raw_out, "latency_s": round(lat,3),
                "correct": gt == parsed["label"],
                "model": MODELS[model_key]["short_name"],
                "strategy": strategy, "condition": condition_tag,
                "parse_ok": parsed["parse_ok"],
            }
            records.append(rec)
            y_true.append(gt); y_pred.append(parsed["label"])
            latencies.append(lat)
            status = "✓" if rec["correct"] else "✗"
            log.info(f"    [{idx+1:3d}/{len(test_set)}] {sid} "
                     f"GT={gt:7s} PRED={parsed['label']:7s} {status} ({lat:.2f}s)")

        # Save raw
        with open(raw_path, "w") as f:
            json.dump({"run_key": run_key, "condition": condition_tag,
                       "model": MODELS[model_key]["display"],
                       "strategy": strategy,
                       "timestamp": datetime.now().isoformat(),
                       "records": records}, f, indent=2)

        m = compute_metrics(y_true, y_pred)
        m["avg_latency_s"] = round(float(np.mean(latencies)),3)
        m.update({"model": MODELS[model_key]["display"],
                  "model_key": model_key, "strategy": strategy,
                  "condition": condition_tag, "run_key": run_key})
        results[strategy] = m
        log.info(f"    → Acc={m['accuracy']:.3f}  MacroF1={m['macro_f1']:.3f}")

    return results


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

COND_COLORS  = {
    "ground_truth":    "#1a1a2e",
    "whisper_base":    "#e94560",
    "whisper_medium":  "#0f3460",
    "whisper_large-v3":"#16213e",
}
COND_MARKERS = {
    "ground_truth":    "D",
    "whisper_base":    "o",
    "whisper_medium":  "s",
    "whisper_large-v3":"^",
}
COND_DISPLAY = {
    "ground_truth":    "Ground Truth (GPT)",
    "whisper_base":    "Whisper-base",
    "whisper_medium":  "Whisper-medium",
    "whisper_large-v3":"Whisper-large-v3",
}
MODEL_COLORS = {"qwen":"#2166AC","mistral":"#762A83","gemma":"#1B7837",
                "gpt-4o":"#B35806","gpt-5.4":"#000000"}
MODEL_NAMES  = {"qwen":"Qwen 2.5-7B","mistral":"Mistral-7B","gemma":"Gemma-2-9B",
                "gpt-4o":"GPT-4o","gpt-5.4":"GPT-5.4"}
MODEL_MARKERS = {"qwen":"o","mistral":"s","gemma":"^","gpt-4o":"D","gpt-5.4":"*"}
STRAT_NAMES  = {"zero_shot":"Zero-shot","one_shot":"One-shot","few_shot":"Few-shot"}

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


def plot_asr_comparison(all_metrics, out_path):
    """Line plot: x = Whisper condition (GT, base, medium, large),"""
    conditions = ["ground_truth","whisper_base","whisper_medium","whisper_large-v3"]
    cond_labs  = ["GT\n(text)","Whisper\nbase","Whisper\nmedium","Whisper\nlarge-v3"]
    x = np.arange(len(conditions))

    with mpl.rc_context(ICML_RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)

        for ci, strategy in enumerate(STRATEGIES):
            ax = axes[ci]
            for mk in ["qwen","mistral","gemma"]:
                ys = []
                for cond in conditions:
                    key = f"{mk}_{strategy}_{cond}"
                    ys.append(all_metrics.get(key, {}).get("macro_f1", None))
                y = [v if v is not None else float("nan") for v in ys]
                ax.plot(x, y, color=MODEL_COLORS[mk],
                        marker=COND_MARKERS["ground_truth"],
                        markersize=4, linewidth=1.1, label=MODEL_NAMES[mk])
                for xi, yi in zip(x, y):
                    if not np.isnan(yi):
                        ax.annotate(f"{yi:.2f}", (xi, yi),
                                    xytext=(0, 5), textcoords="offset points",
                                    ha="center", fontsize=5.5,
                                    color=MODEL_COLORS[mk])

            ax.set_xticks(x)
            ax.set_xticklabels(cond_labs, fontsize=6.5)
            ax.set_title(STRAT_NAMES[strategy], fontsize=8.5, pad=3)
            ax.grid(axis="y", color="#CCCCCC", linewidth=0.4, linestyle="--")
            ax.set_ylim(0.35, 0.85)
            if ci == 0:
                ax.set_ylabel("Macro-averaged F$_1$")
            if ci == 1:
                ax.legend(loc="lower center", ncol=3, frameon=True,
                          framealpha=0.9, edgecolor="#CCCCCC",
                          bbox_to_anchor=(0.5, -0.35), fontsize=6.5)

        fig.suptitle("ASR quality ablation: macro-F$_1$ vs. transcript source",
                     fontsize=9.5, y=1.02, x=0.02, ha="left")
        save_fig(fig, out_path)


def plot_asr_perclass(all_metrics, out_path):
    """Grouped bars per condition, split by class F1, for zero-shot only."""
    conditions  = ["ground_truth","whisper_base","whisper_medium","whisper_large-v3"]
    cond_colors = ["#252525","#D6604D","#4393C3","#1B7837"]
    bw  = 0.18
    x   = np.arange(3)   # 3 models

    with mpl.rc_context(ICML_RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)

        for li, lbl in enumerate(LABELS):
            ax = axes[li]
            for ci, cond in enumerate(conditions):
                vals = []
                for mk in ["qwen","mistral","gemma"]:
                    key = f"{mk}_zero_shot_{cond}"
                    vals.append(all_metrics.get(key,{}).get(
                        "per_class",{}).get(lbl,{}).get("f1",0))
                xpos = x + (ci - 1.5) * bw
                ax.bar(xpos, vals, bw, color=cond_colors[ci],
                       edgecolor="white", linewidth=0.3,
                       label=COND_DISPLAY[cond], zorder=3)

            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_NAMES[m] for m in ["qwen","mistral","gemma"]],
                               fontsize=6.5)
            ax.set_title(lbl.capitalize(), fontsize=8.5)
            ax.grid(axis="y", color="#CCCCCC", linewidth=0.4, linestyle="--")
            ax.set_ylim(0, 1.05)
            if li == 0:
                ax.set_ylabel("F$_1$ score (zero-shot)")
            if li == 1:
                ax.legend(loc="upper center", ncol=2, frameon=True,
                          framealpha=0.9, edgecolor="#CCCCCC",
                          bbox_to_anchor=(0.5, -0.28), fontsize=6.0)

        fig.suptitle("Per-class F$_1$: ASR quality ablation (zero-shot)",
                     fontsize=9.5, y=1.02, x=0.02, ha="left")
        save_fig(fig, out_path)


def make_asr_table(all_metrics):
    conditions = ["ground_truth","whisper_base","whisper_medium","whisper_large-v3"]
    cond_display = {
        "ground_truth":    r"GT (text)",
        "whisper_base":    r"Whisper-base",
        "whisper_medium":  r"Whisper-medium",
        "whisper_large-v3":r"Whisper-large-v3",
    }
    cond_heads = " & ".join(
        r"\textbf{" + cond_display[c] + r"}" for c in conditions)

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{ASR quality ablation: macro-F$_1$ across transcript sources "
        r"and prompting strategies. GT = ground-truth GPT-generated text (no ASR).}",
        r"\label{tab:asr_ablation}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll" + "c" * len(conditions) + r"}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Strategy} & " + cond_heads + r" \\",
        r"\midrule",
    ]

    for mk in ["qwen","mistral","gemma"]:
        for si, st in enumerate(STRATEGIES):
            model_cell = (r"\multirow{3}{*}{" + MODEL_NAMES[mk] + r"}"
                          if si == 0 else "")
            # Find best value across conditions for this model+strategy
            vals = {c: all_metrics.get(f"{mk}_{st}_{c}",{}).get("macro_f1",0)
                    for c in conditions}
            best = max(vals.values())
            cells = []
            for c in conditions:
                v = vals[c]
                s = f"{v:.3f}"
                cells.append(r"\textbf{" + s + r"}" if abs(v-best) < 1e-6 else s)
            newline = r"\\"
            lines.append(
                f"  {model_cell} & {STRAT_NAMES[st]} & "
                + " & ".join(cells) + f" {newline}")
        lines.append(r"  \midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    import mpl_toolkits  # noqa — ensure matplotlib imports work
    global mpl
    import matplotlib as mpl

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    required=True)
    ap.add_argument("--audio-dir",  required=True,
                    help="Path to dataset/scenarios/ directory")
    ap.add_argument("--gt-metrics", required=True,
                    help="Path to existing results/metrics/all_metrics.json")
    ap.add_argument("--out",        default="results/ablation_whisper")
    ap.add_argument("--whisper-sizes", nargs="+", default=WHISPER_SIZES,
                    choices=WHISPER_SIZES)
    ap.add_argument("--models",     nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--skip-transcription", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--quantize",   default=None, choices=["4bit","8bit"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out)

    log.info("=" * 60)
    log.info("Ablation 1 — Whisper ASR Quality")
    log.info(f"  Whisper sizes : {args.whisper_sizes}")
    log.info(f"  Models        : {args.models}")
    log.info(f"  Output        : {out}/")
    log.info("=" * 60)

    # Load dataset
    with open(args.dataset) as f:
        data = json.load(f)
    all_scenarios = data["scenarios"]
    icl_ids  = set(ICL_EXAMPLE_IDS.values())
    icl_pool = {s["scenario_id"]: s for s in all_scenarios
                if s["scenario_id"] in icl_ids}
    test_set = [s for s in all_scenarios
                if s["scenario_id"] not in icl_ids]
    icl_ordered = [
        icl_pool[ICL_EXAMPLE_IDS["nominal_1"]],
        icl_pool[ICL_EXAMPLE_IDS["nominal_2"]],
        icl_pool[ICL_EXAMPLE_IDS["warning_1"]],
        icl_pool[ICL_EXAMPLE_IDS["warning_2"]],
        icl_pool[ICL_EXAMPLE_IDS["hazard_1"]],
        icl_pool[ICL_EXAMPLE_IDS["hazard_2"]],
    ]
    log.info(f"Test set: {len(test_set)} | ICL pool: {len(icl_pool)}")

    # Load existing GT metrics
    with open(args.gt_metrics) as f:
        gt_metrics_raw = json.load(f)

    # Re-key GT metrics to include condition tag
    all_metrics = {}
    for key, val in gt_metrics_raw.items():
        # key is e.g. "qwen_zero_shot" → remap to "qwen_zero_shot_ground_truth"
        new_key = f"{key}_ground_truth"
        val_copy = dict(val)
        val_copy["condition"] = "ground_truth"
        val_copy["run_key"]   = new_key
        all_metrics[new_key]  = val_copy

    if args.plots_only:
        met_path = out / "metrics" / "whisper_all_metrics.json"
        if met_path.exists():
            with open(met_path) as f:
                all_metrics = json.load(f)
        _save_all_outputs(all_metrics, out, log)
        return

    tx_dir = out / "transcripts"

    transcripts = {}   # size -> {sid: text}
    if not args.skip_transcription:
        for wsize in args.whisper_sizes:
            log.info(f"\nTranscribing with Whisper-{wsize}...")
            transcripts[wsize] = run_transcription_batch(
                all_scenarios, args.audio_dir, wsize, tx_dir, log)
    else:
        log.info("Skipping transcription (--skip-transcription set)")
        for wsize in args.whisper_sizes:
            transcripts[wsize] = {}
            for sc in all_scenarios:
                sid  = sc["scenario_id"]
                path = tx_dir / f"{sid}_{wsize}.txt"
                if path.exists():
                    transcripts[wsize][sid] = path.read_text(encoding="utf-8")

    import torch
    for model_key in args.models:
        log.info(f"\n{'='*60}\nModel: {MODELS[model_key]['display']}")
        model, tok = load_model(model_key, log)

        for wsize in args.whisper_sizes:
            cond_tag = f"whisper_{wsize}"
            log.info(f"  Condition: {cond_tag}")
            tx_map   = {sid: transcripts[wsize].get(sid,"")
                        for sid in [s["scenario_id"] for s in all_scenarios]}
            run_res  = evaluate_on_transcripts(
                model_key, model, tok, test_set, icl_ordered,
                tx_map, cond_tag, out, log)
            for strategy, m in run_res.items():
                all_metrics[m["run_key"]] = m

        if not MODELS[model_key].get("is_openai"):
            del model, tok
            torch.cuda.empty_cache()
            log.info(f"  GPU freed after {MODELS[model_key]['display']}")
        else:
            log.info(f"  Done with {MODELS[model_key]['display']}")

    _save_all_outputs(all_metrics, out, log)


def _save_all_outputs(all_metrics, out, log):
    # Metrics JSON
    met_dir = out / "metrics"
    met_dir.mkdir(exist_ok=True)
    with open(met_dir / "whisper_all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # CSV
    rows = []
    for key, m in all_metrics.items():
        rows.append({
            "run_key":   key,
            "model":     m.get("model",""),
            "strategy":  m.get("strategy",""),
            "condition": m.get("condition",""),
            "accuracy":  m.get("accuracy",0),
            "macro_f1":  m.get("macro_f1",0),
            "latency_s": m.get("avg_latency_s",0),
        })
    import csv
    with open(met_dir / "whisper_summary.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # Figures
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_asr_comparison(all_metrics, fig_dir / "fig_asr_comparison.pdf")
    plot_asr_perclass(all_metrics,   fig_dir / "fig_asr_perclass_f1.pdf")

    # Table
    tab_dir = out / "tables"
    tab_dir.mkdir(exist_ok=True)
    (tab_dir / "table_asr_ablation.tex").write_text(make_asr_table(all_metrics))

    log.info(f"\nAll outputs saved to {out}/")
    log.info("Run summary:")
    log.info(f"  {'Run key':<45} {'Acc':>6} {'F1':>6}")
    log.info(f"  {'-'*60}")
    for key, m in sorted(all_metrics.items(),
                         key=lambda x: x[1].get("macro_f1",0), reverse=True):
        log.info(f"  {key:<45} {m.get('accuracy',0):>6.3f} {m.get('macro_f1',0):>6.3f}")


if __name__ == "__main__":
    main()