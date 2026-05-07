#!/usr/bin/env python3
"""Ablation 2 — Audio Noise Robustness"""

import os, sys, json, re, time, copy, argparse, logging, csv
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

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

# (copy relevant constants here to keep script self-contained)

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
LABELS      = ["nominal", "warning", "hazard"]
NOISE_LEVELS = [5, 10, 25, 50, 75]   # NSR percent
WHISPER_SIZE = "large-v3"

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

Note: The transcript may contain transcription artifacts from noisy radio audio.

Respond with ONLY this JSON structure:
{
  "label": "<nominal|warning|hazard>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence>"
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
def translate_phonetic(t):
    return _NATO_PAT.sub(lambda m: NATO[m.group(0).lower()], t)


def setup_logging(out_dir):
    log = logging.getLogger("noise_abl")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%H:%M:%S")
    log.addHandler(logging.StreamHandler())
    fh = logging.FileHandler(out_dir / "ablation_noise.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    for h in log.handlers:
        h.setFormatter(fmt)
    return log


def add_noise_to_audio(audio_path: Path, nsr_pct: float,
                       output_path: Path) -> Path:
    """Add AWGN to an MP3 file at the given noise-to-signal ratio (percent)."""
    from pydub import AudioSegment
    import io

    audio    = AudioSegment.from_file(str(audio_path), format="mp3")
    samples  = np.array(audio.get_array_of_samples(), dtype=np.float32)

    # Compute signal RMS
    rms_signal = np.sqrt(np.mean(samples ** 2))
    if rms_signal < 1e-6:
        # Silent audio — just copy
        audio.export(str(output_path), format="mp3")
        return output_path

    # Generate AWGN
    nsr        = nsr_pct / 100.0
    rms_noise  = rms_signal * nsr
    noise      = np.random.randn(len(samples)).astype(np.float32) * rms_noise

    noisy      = samples + noise
    # Clip to int16 range
    noisy      = np.clip(noisy, -32768, 32767).astype(np.int16)

    # Reconstruct AudioSegment
    noisy_seg  = audio._spawn(noisy.tobytes())
    noisy_seg.export(str(output_path), format="mp3", bitrate="128k")
    return output_path


def _whisper_device():
    """Safe CUDA probe — falls back to CPU if libcublas is missing."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu", "int8"
        import ctypes
        ctypes.CDLL("libcublas.so.12")
        return "cuda", "float16"
    except Exception:
        return "cpu", "int8"


def transcribe_audio(audio_path: Path, whisper_size: str) -> str:
    """Transcribe audio file using faster-whisper. Returns NATO-translated text."""
    from faster_whisper import WhisperModel

    device, compute_type = _whisper_device()
    model        = WhisperModel(whisper_size, device=device,
                                compute_type=compute_type)

    segments, _ = model.transcribe(
        str(audio_path), beam_size=5, word_timestamps=True,
        vad_filter=True,
        vad_parameters={"threshold": 0.5,      # lower threshold for noisy audio
                         "min_speech_duration_ms": 200,
                         "max_speech_duration_s": float("inf"),
                         "min_silence_duration_ms": 150,
                         "speech_pad_ms": 500})

    word_segs = []
    for seg in segments:
        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                if w.word.strip():
                    word_segs.append({"text": w.word.strip(),
                                      "start": w.start, "end": w.end})

    # Merge into sentence segments
    merged, cur = [], None
    for s in word_segs:
        txt = s["text"].strip()
        if not txt:
            continue
        if cur is None:
            cur = {"text": txt, "start": s["start"], "end": s["end"]}
            continue
        if s["start"] - cur["end"] < 2.0 and len(cur["text"]+txt) < 80:
            cur["text"] = (cur["text"] + " " + txt).strip()
            cur["end"]  = s["end"]
        else:
            merged.append(copy.deepcopy(cur))
            cur = {"text": txt, "start": s["start"], "end": s["end"]}
    if cur:
        merged.append(cur)

    try:
        import srt as srt_lib
        srt_list = [
            srt_lib.Subtitle(index=i,
                             start=timedelta(seconds=v["start"]),
                             end=timedelta(seconds=v["end"]),
                             content=v["text"].strip())
            for i, v in enumerate(merged)
        ]
        raw = srt_lib.compose(srt_list)
    except ImportError:
        raw = "\n".join(v["text"] for v in merged)

    del model
    if device == "cuda":
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
    return translate_phonetic(raw)


def format_input(scenario, transcript_text):
    metar = scenario["metar"]
    mb = (f"METAR: {metar['raw']}\n"
          f"Conditions: {metar['description']} | Wind: {metar['wind_kt']} kt")
    return (f"--- WEATHER ---\n{mb}\n\n"
            f"--- CTAF TRANSCRIPT (from noisy audio) ---\n{transcript_text}\n\n"
            f"Classify the safety level:")

def build_zero_shot(scenario, transcript_text, model_key):
    NO_SYS  = {"gemma"}
    use_sys = model_key not in NO_SYS
    content = format_input(scenario, transcript_text)
    if use_sys:
        return [{"role":"system","content":SYSTEM_PROMPT},
                {"role":"user",  "content":content}]
    else:
        return [{"role":"user",
                 "content":f"{SYSTEM_PROMPT}\n\n{content}"}]

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
            label = _SYNONYMS.get(label, label)
            if label not in LABELS:
                for lbl in ["hazard","warning","nominal"]:
                    if lbl in clean.lower():
                        label = lbl; break
            return {"label":label,
                    "confidence":float(obj.get("confidence",0.5)),
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
              dtype=torch.float16, token=os.environ.get("HF_TOKEN"))
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
    elif isinstance(chat_out, dict):
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
    cm  = np.zeros((3,3), dtype=int)
    l2i = {l:i for i,l in enumerate(LABELS)}
    for yt,yp in zip(y_true, y_pred):
        ti,pi = l2i.get(yt,-1), l2i.get(yp,-1)
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
LABEL_COLORS = {"nominal":"#4DAC26","warning":"#E08214","hazard":"#D6604D"}

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


def plot_noise_f1_curve(all_metrics, noise_levels, gt_metrics, out_path):
    """Main noise ablation figure: x = NSR%, y = macro F1."""
    x = noise_levels

    with mpl.rc_context(ICML_RC):
        fig, ax = plt.subplots(figsize=(3.8, 2.8))

        for mk in ["qwen","mistral","gemma"]:
            ys = [all_metrics.get(f"{mk}_zero_shot_noise_{lvl}",{})
                             .get("macro_f1", float("nan"))
                  for lvl in noise_levels]

            ax.plot(x, ys,
                    color=MODEL_COLORS[mk],
                    marker=MODEL_MARKERS[mk],
                    markersize=5, linewidth=1.2,
                    label=MODEL_NAMES[mk], zorder=4)

            # GT text baseline (dotted)
            gt_f1 = gt_metrics.get(f"{mk}_zero_shot",{}).get("macro_f1")
            if gt_f1:
                ax.axhline(gt_f1, color=MODEL_COLORS[mk],
                           linewidth=0.7, linestyle=":",
                           alpha=0.6, zorder=3)

        ax.set_xlabel("Noise-to-signal ratio (\\%)")
        ax.set_ylabel("Macro-averaged F$_1$")
        ax.set_xticks(noise_levels)
        ax.set_xticklabels([f"{l}\\%" for l in noise_levels])
        ax.set_ylim(0.0, 0.85)
        ax.grid(axis="y", color="#CCCCCC", linewidth=0.4, linestyle="--")

        # Annotations
        ax.text(0.98, 0.97,
                "Dotted lines = GT text baseline",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.0, color="#666666", style="italic")

        ax.legend(loc="upper right", frameon=True,
                  framealpha=0.9, edgecolor="#CCCCCC")
        ax.set_title("Macro-F$_1$ vs.~audio noise level\n"
                     "(zero-shot, Whisper large-v3 transcription)",
                     pad=4, loc="left", fontsize=8.5)
        save_fig(fig, out_path)


def plot_noise_perclass(all_metrics, noise_levels, out_path):
    """One panel per safety class: line per model, x = noise level."""
    with mpl.rc_context(ICML_RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)

        for li, lbl in enumerate(LABELS):
            ax = axes[li]
            for mk in ["qwen","mistral","gemma"]:
                ys = [all_metrics.get(f"{mk}_zero_shot_noise_{lvl}",{})
                                 .get("per_class",{}).get(lbl,{})
                                 .get("f1", float("nan"))
                      for lvl in noise_levels]
                ax.plot(noise_levels, ys,
                        color=MODEL_COLORS[mk],
                        marker=MODEL_MARKERS[mk],
                        markersize=4, linewidth=1.1,
                        label=MODEL_NAMES[mk], zorder=4)

            ax.set_xticks(noise_levels)
            ax.set_xticklabels([f"{l}\\%" for l in noise_levels],
                               fontsize=6.5)
            ax.set_title(lbl.capitalize(), fontsize=8.5)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", color="#CCCCCC", linewidth=0.4, linestyle="--")
            if li == 0:
                ax.set_ylabel("F$_1$ score")
            ax.set_xlabel("NSR (\\%)", fontsize=7.5)
            if li == 1:
                ax.legend(loc="upper center", ncol=3, frameon=True,
                          framealpha=0.9, edgecolor="#CCCCCC",
                          bbox_to_anchor=(0.5, -0.28), fontsize=6.5)

        fig.suptitle("Per-class F$_1$ vs.~audio noise (zero-shot)",
                     fontsize=9.5, y=1.02, x=0.02, ha="left")
        save_fig(fig, out_path)


def make_noise_table(all_metrics, noise_levels, gt_metrics):
    level_heads = " & ".join(f"NSR {l}\\%" for l in noise_levels)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Audio noise robustness: macro-F$_1$ at increasing noise-to-signal "
        r"ratio (NSR). GT = ground-truth text input (no ASR). "
        r"All results use zero-shot prompting with Whisper large-v3 transcription.}",
        r"\label{tab:noise_ablation}",
        r"\begin{tabular}{llc" + "c"*len(noise_levels) + r"}",
        r"\toprule",
        r"\textbf{Model} & \textbf{GT} & " + level_heads + r" \\",
        r"\midrule",
    ]
    for mk in ["qwen","mistral","gemma"]:
        gt_f1 = gt_metrics.get(f"{mk}_zero_shot",{}).get("macro_f1",0)
        noise_vals = [all_metrics.get(f"{mk}_zero_shot_noise_{l}",{})
                                 .get("macro_f1",0) for l in noise_levels]
        all_vals  = [gt_f1] + noise_vals
        best      = max(all_vals)
        def b(v):
            s = f"{v:.3f}"
            return r"\textbf{" + s + r"}" if abs(v-best)<1e-6 else s
        newline = r"\\"
        lines.append(
            f"  {MODEL_NAMES[mk]} & {b(gt_f1)} & "
            + " & ".join(b(v) for v in noise_vals) + f" {newline}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",     required=True)
    ap.add_argument("--audio-dir",   required=True)
    ap.add_argument("--gt-metrics",  required=True,
                    help="Path to results/metrics/all_metrics.json")
    ap.add_argument("--out",         default="results/ablation_noise")
    ap.add_argument("--noise-levels",nargs="+", type=int,
                    default=NOISE_LEVELS)
    ap.add_argument("--models",      nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--whisper-size",default=WHISPER_SIZE)
    ap.add_argument("--plots-only",  action="store_true")
    ap.add_argument("--purge-cache", action="store_true",
                    help="Delete all cached noisy audio and transcripts and start fresh")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--force-cpu-whisper", action="store_true",
                    help="Force Whisper to run on CPU (use if libcublas missing)")
    args = ap.parse_args()

    if args.force_cpu_whisper:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Purge stale cache if requested
    if args.purge_cache:
        import shutil
        for subdir in ["noisy_audio", "transcripts", "raw"]:
            p = out / subdir
            if p.exists():
                shutil.rmtree(p)
                print(f"  Purged {p}")
        print("Cache cleared. Re-run without --purge-cache.")
    log = setup_logging(out)

    dev, ctype = _whisper_device()
    log.info("=" * 60)
    log.info("Ablation 2 — Audio Noise Robustness")
    log.info(f"  Noise levels  : {args.noise_levels} NSR%")
    log.info(f"  Whisper size  : {args.whisper_size}")
    log.info(f"  Whisper device: {dev} / {ctype}")
    log.info(f"  Models        : {args.models}")
    log.info("=" * 60)

    with open(args.dataset) as f:
        data = json.load(f)
    all_scenarios = data["scenarios"]
    icl_ids  = set(ICL_EXAMPLE_IDS.values())
    test_set = [s for s in all_scenarios
                if s["scenario_id"] not in icl_ids]

    with open(args.gt_metrics) as f:
        gt_metrics = json.load(f)

    all_metrics = {}
    audio_dir   = Path(args.audio_dir)
    noisy_dir   = out / "noisy_audio"
    tx_dir      = out / "transcripts"
    noisy_dir.mkdir(exist_ok=True)
    tx_dir.mkdir(exist_ok=True)

    if args.plots_only:
        met_path = out / "metrics" / "noise_all_metrics.json"
        if met_path.exists():
            with open(met_path) as f:
                all_metrics = json.load(f)
        _save_outputs(all_metrics, args.noise_levels, gt_metrics, out, log)
        return

    # First: verify ffmpeg/pydub works before processing 500 files
    log.info("\nChecking pydub/ffmpeg availability...")
    try:
        from pydub import AudioSegment
        # Try loading the first available audio file as a smoke test
        test_audio = next(
            (audio_dir / sc["scenario_id"] / "audio.mp3"
             for sc in all_scenarios
             if (audio_dir / sc["scenario_id"] / "audio.mp3").exists()),
            None
        )
        if test_audio:
            _ = AudioSegment.from_file(str(test_audio), format="mp3")
            log.info(f"  pydub OK — loaded {test_audio.name}")
        else:
            raise FileNotFoundError("No audio files found in audio-dir")
    except Exception as e:
        log.error(f"  pydub/ffmpeg check FAILED: {e}")
        log.error("  Install ffmpeg: sudo apt install ffmpeg")
        log.error("  Then re-run. Aborting.")
        sys.exit(1)

    # Purge any empty cached transcripts from a previous failed run
    purged = 0
    for cache_file in tx_dir.glob("*.txt"):
        if cache_file.stat().st_size == 0:
            cache_file.unlink()
            purged += 1
    if purged:
        log.info(f"  Purged {purged} empty transcript cache files")

    transcripts = defaultdict(dict)   # noise_level -> {sid: text}

    for nsr in args.noise_levels:
        log.info(f"\nNoise level NSR={nsr}%  ({len(all_scenarios)} scenarios)")
        ok_count = fail_count = cached_count = 0

        for sc in all_scenarios:
            sid        = sc["scenario_id"]
            tx_cache   = tx_dir / f"{sid}_noise_{nsr}.txt"

            # Resume: only use cache if non-empty
            if tx_cache.exists() and tx_cache.stat().st_size > 10:
                transcripts[nsr][sid] = tx_cache.read_text(encoding="utf-8")
                cached_count += 1
                continue

            orig_audio  = audio_dir / sid / "audio.mp3"
            noisy_audio = noisy_dir / f"{sid}_noise_{nsr}.mp3"

            if not orig_audio.exists():
                log.warning(f"  {sid}: audio.mp3 not found at {orig_audio}")
                transcripts[nsr][sid] = ""
                fail_count += 1
                continue

            # Always regenerate noisy audio (don't trust stale empty files)
            if not noisy_audio.exists() or noisy_audio.stat().st_size < 1000:
                try:
                    add_noise_to_audio(orig_audio, float(nsr), noisy_audio)
                    log.debug(f"  {sid} NSR{nsr}%: noise injected "
                              f"({noisy_audio.stat().st_size//1024} KB)")
                except Exception as e:
                    log.error(f"  {sid} noise inject FAILED: {e}")
                    transcripts[nsr][sid] = ""
                    fail_count += 1
                    continue

            try:
                text = transcribe_audio(noisy_audio, args.whisper_size)
                if not text.strip():
                    raise ValueError("Whisper returned empty transcript")
                tx_cache.write_text(text, encoding="utf-8")
                transcripts[nsr][sid] = text
                n_words = len(text.split())
                log.info(f"  NSR{nsr}% {sid}: {n_words} words transcribed")
                ok_count += 1
            except Exception as e:
                log.error(f"  {sid} NSR{nsr}% transcription FAILED: {e}")
                # Do NOT cache empty result — allow retry on next run
                transcripts[nsr][sid] = ""
                fail_count += 1

        log.info(f"  NSR={nsr}%: OK={ok_count} cached={cached_count} "
                 f"failed={fail_count}")
        if fail_count == len(all_scenarios):
            log.error(f"  ALL transcriptions failed at NSR={nsr}%. "
                      f"Check ffmpeg and audio paths. Aborting.")
            sys.exit(1)

        # Sanity check: verify transcripts are actually different across scenarios
        unique_texts = len(set(transcripts[nsr].values()))
        log.info(f"  NSR={nsr}%: {unique_texts} unique transcripts "
                 f"(expected ~{len(all_scenarios)})")

    import torch
    for model_key in args.models:
        log.info(f"\n{'='*60}\nModel: {MODELS[model_key]['display']}")
        model, tok = load_model(model_key, log)

        for nsr in args.noise_levels:
            cond_tag = f"noise_{nsr}"
            run_key  = f"{model_key}_zero_shot_{cond_tag}"
            raw_path = out / "raw" / f"{run_key}.json"

            # Guard: skip if all transcripts for this noise level are empty
            non_empty = sum(1 for sid, tx in transcripts[nsr].items()
                           if tx.strip())
            if non_empty == 0:
                log.error(f"  {run_key}: ALL transcripts empty — skipping. "
                          f"Re-run transcription step first.")
                continue
            log.info(f"  {cond_tag}: {non_empty}/{len(transcripts[nsr])} "
                     f"non-empty transcripts")
            cond_tag = f"noise_{nsr}"
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
                           "noise_level_pct":nsr})
                all_metrics[run_key] = m
                continue

            log.info(f"  NSR={nsr}%  ({len(test_set)} scenarios)")
            records, y_true, y_pred, latencies = [], [], [], []

            for idx, sc in enumerate(test_set):
                sid  = sc["scenario_id"]
                tx   = transcripts[nsr].get(sid, "")
                msgs = build_zero_shot(sc, tx, model_key)
                raw_out, lat = run_inference(model, tok, msgs,
                                            MODELS[model_key])
                parsed = parse_response(raw_out)
                gt     = sc["label"]
                rec    = {
                    "scenario_id": sid,
                    "hazard_type": sc["hazard_type"],
                    "ground_truth": gt,
                    "predicted": parsed["label"],
                    "confidence": parsed["confidence"],
                    "reasoning": parsed["reasoning"],
                    "raw_output": raw_out,
                    "latency_s": round(lat, 3),
                    "correct": gt == parsed["label"],
                    "model": MODELS[model_key]["short_name"],
                    "condition": cond_tag,
                    "noise_level_pct": nsr,
                    "parse_ok": parsed["parse_ok"],
                    "transcript_words": len(tx.split()) if tx else 0,
                }
                records.append(rec)
                y_true.append(gt); y_pred.append(parsed["label"])
                latencies.append(lat)
                status = "✓" if rec["correct"] else "✗"
                log.info(f"    [{idx+1:3d}/{len(test_set)}] {sid} "
                         f"GT={gt:7s} PRED={parsed['label']:7s} {status} "
                         f"({lat:.2f}s, {rec['transcript_words']}w)")

            with open(raw_path, "w") as f:
                json.dump({"run_key":run_key,"condition":cond_tag,
                           "noise_level_pct":nsr,
                           "model":MODELS[model_key]["display"],
                           "timestamp":datetime.now().isoformat(),
                           "records":records}, f, indent=2)

            m = compute_metrics(y_true, y_pred)
            m["avg_latency_s"] = round(float(np.mean(latencies)),3)
            m.update({"model":MODELS[model_key]["display"],
                       "model_key":model_key,"strategy":"zero_shot",
                       "condition":cond_tag,"run_key":run_key,
                       "noise_level_pct":nsr})
            all_metrics[run_key] = m
            log.info(f"    → Acc={m['accuracy']:.3f}  MacroF1={m['macro_f1']:.3f}")

        if not MODELS[model_key].get("is_openai"):
            del model, tok
            torch.cuda.empty_cache()

    _save_outputs(all_metrics, args.noise_levels, gt_metrics, out, log)


def _save_outputs(all_metrics, noise_levels, gt_metrics, out, log):
    met_dir = out / "metrics"
    met_dir.mkdir(exist_ok=True)
    with open(met_dir / "noise_all_metrics.json","w") as f:
        json.dump(all_metrics, f, indent=2)

    rows = []
    for key, m in all_metrics.items():
        rows.append({"run_key":key,"model":m.get("model",""),
                     "noise_pct":m.get("noise_level_pct",""),
                     "accuracy":m.get("accuracy",0),
                     "macro_f1":m.get("macro_f1",0)})
    with open(met_dir/"noise_summary.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_noise_f1_curve(all_metrics, noise_levels, gt_metrics,
                        fig_dir/"fig_noise_f1_curve.pdf")
    plot_noise_perclass(all_metrics, noise_levels,
                        fig_dir/"fig_noise_perclass.pdf")

    tab_dir = out / "tables"
    tab_dir.mkdir(exist_ok=True)
    (tab_dir/"table_noise_ablation.tex").write_text(
        make_noise_table(all_metrics, noise_levels, gt_metrics))

    log.info(f"\nSaved all noise ablation outputs to {out}/")


if __name__ == "__main__":
    main()