#!/usr/bin/env python3
"""CTAF-KHAF Open-Source LLM Evaluation Pipeline"""

import os
import sys
import json
import time
import re
import copy
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    from openai_inference import (
        OPENAI_MODELS, OPENAI_MODEL_KEYS,
        is_openai_model, load_openai_client, run_openai_inference,
        run_openai_inference_cot, openai_cost_estimate,
        score_classes_openai,
    )
    _OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_MODELS = {}; OPENAI_MODEL_KEYS = []
    _OPENAI_AVAILABLE = False
    def is_openai_model(k): return False
    def score_classes_openai(*a, **kw): return {}
    import warnings
    warnings.warn(
        "openai_inference.py not found in the current directory. "
        "Copy it alongside run_experiments.py to enable --closed-source-model.",
        stacklevel=1
    )

try:
    from claude_inference import (
        CLAUDE_MODELS, CLAUDE_MODEL_KEYS,
        is_claude_model, load_claude_client, run_claude_inference,
        run_claude_inference_cot, claude_cost_estimate,
    )
    _CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_MODELS = {}; CLAUDE_MODEL_KEYS = []
    _CLAUDE_AVAILABLE = False
    def is_claude_model(k): return False


def setup_logging(out_dir: Path) -> logging.Logger:
    log = logging.getLogger("ctaf_eval")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / "experiment.log")
    fh.setFormatter(fmt)
    log.addHandler(ch)
    log.addHandler(fh)
    return log


MODELS = {
    "qwen": {
        "hf_id":       "Qwen/Qwen2.5-7B-Instruct",
        "short_name":  "Qwen2.5-7B",
        "display":     "Qwen 2.5-7B-Instruct",
        "vram_gb":     15,
        "chat_format": "chatml",       # uses apply_chat_template
        "max_new_tokens": 120,
        "temperature":    0.0,         # greedy for reproducibility
    },
    "mistral": {
        "hf_id":       "mistralai/Mistral-7B-Instruct-v0.3",
        "short_name":  "Mistral-7B",
        "display":     "Mistral-7B-Instruct-v0.3",
        "vram_gb":     14,
        "chat_format": "mistral",
        "max_new_tokens": 120,
        "temperature":    0.0,
    },
    "gemma": {
        "hf_id":       "google/gemma-2-9b-it",
        "short_name":  "Gemma-2-9B",
        "display":     "Gemma-2-9B-IT",
        "vram_gb":     18,
        "chat_format": "gemma",
        "max_new_tokens": 120,
        "temperature":    0.0,
    },
}

STRATEGIES = ["zero_shot", "one_shot", "few_shot"]
LABELS     = ["nominal", "warning", "hazard"]

# Binary-mode labels (collapses warning + hazard -> danger).
LABELS_BINARY = ["nominal", "danger"]

def collapse_label(lbl: str) -> str:
    """Map 3-class label to binary: warning|hazard -> danger; nominal stays."""
    return "nominal" if lbl == "nominal" else "danger"


# Step-1 prompt appended to the user turn when CoT is enabled.
# Asks the model to reason before committing to a label.
COT_THINK_INSTRUCTION = """
Before classifying, reason step by step:
1. What aircraft are present and what are their positions/intentions?
2. Are there any communication gaps (missing calls, NORDO traffic)?
3. Is there a conflict? If so, is it POTENTIAL (still time to resolve) or IMMINENT (collision NOW)?
4. Apply the warning vs hazard test: would a CTAF advisory say "IMMEDIATELY" or "SAFETY ALERT"?

Write your reasoning, then end with the JSON on its own line."""

# Step-2 prompt sent as a follow-up user turn to extract the final JSON
# after the model has produced its chain-of-thought.
COT_EXTRACT_INSTRUCTION = (
    "Based on your reasoning above, provide ONLY the final JSON classification "
    "(no other text):\n"
    "{\n"
    '  "label": "<nominal|warning|hazard>",\n'
    '  "confidence": <0.0-1.0>,\n'
    '  "reasoning": "<one sentence summary of the key safety factor>"\n'
    "}"
)

# Max new tokens for CoT — needs much more room for reasoning
COT_MAX_NEW_TOKENS = 512


SYSTEM_PROMPT_BINARY = """You are an automated aviation safety monitoring system for Half Moon Bay Airport (KHAF), a non-towered airport near San Francisco, California.

Your task is to analyze CTAF (Common Traffic Advisory Frequency) radio communications at KHAF and classify the safety status of the current traffic situation.

You will be given:
1. METAR weather data for KHAF
2. A CTAF radio transcript (SRT format with timestamps)

━━━ CLASSIFICATION RULES ━━━

Classify as exactly ONE of: nominal | danger

NOMINAL — All is well.
  • All required position calls are present (crosswind, downwind, base, final)
  • Traffic is sequenced and separated with no conflicts
  • Weather is VMC and appropriate for operations
  • Single aircraft announcing each leg, no other traffic

DANGER — Any potential or imminent safety issue.
  Use danger whenever there is ANY conflict, communication gap, or unsafe condition:
  • Communication gaps: missing position calls, NORDO traffic, delayed announcements
  • Pattern conflicts: converging traffic, wrong-runway calls, improper entries
  • Active conflicts: simultaneous final, runway incursions, mid-air risk
  • Weather mismatches: VFR pilot inadvertently in IMC
  • Late or omitted go-around announcements
  • Any situation a CTAF advisory would flag as caution, alert, or emergency

━━━ CTAF RULES (FAA AC 90-66C) ━━━
- Pilots MUST self-announce: crosswind, downwind, base, final, runway clear
- Straight-in: announce at 10, 5, and 3 NM
- Go-around MUST be announced immediately
- No ATC — pilots are solely responsible for separation

Respond with ONLY this JSON (no other text):
{
  "label": "<nominal|danger>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence: state the specific safety factor>"
}"""

COT_THINK_BINARY = """
Before classifying, reason step by step:
1. What aircraft are present and what are their positions/intentions?
2. Are there any communication gaps (missing calls, NORDO traffic)?
3. Is there any conflict, unsafe condition, or potential collision risk?
4. Would a CTAF advisory flag this for any reason (caution, alert, or emergency)?

Write your reasoning, then end with the JSON on its own line."""

COT_EXTRACT_BINARY = (
    "Based on your reasoning above, provide ONLY the final JSON classification "
    "(no other text):\n"
    "{\n"
    '  "label": "<nominal|danger>",\n'
    '  "confidence": <0.0-1.0>,\n'
    '  "reasoning": "<one sentence summary>"\n'
    "}"
)


def apply_binary_mode():
    """Swap module-level constants to binary classification."""
    global LABELS, SYSTEM_PROMPT, COT_THINK_INSTRUCTION, COT_EXTRACT_INSTRUCTION
    global _LABEL_SYNONYMS
    LABELS = LABELS_BINARY
    SYSTEM_PROMPT = SYSTEM_PROMPT_BINARY
    COT_THINK_INSTRUCTION = COT_THINK_BINARY
    COT_EXTRACT_INSTRUCTION = COT_EXTRACT_BINARY
    _LABEL_SYNONYMS = {
        "safe":     "nominal",  "normal":   "nominal",
        "clear":    "nominal",  "ok":       "nominal",
        "warning":  "danger",   "warn":     "danger",
        "alert":    "danger",   "caution":  "danger",
        "hazard":   "danger",   "critical": "danger",
        "emergency":"danger",   "unsafe":   "danger",
    }

# These 6 scenarios are used as ICL examples and EXCLUDED from the test set.
# Chosen to be: short, clear, one per label × 2 diversity types.

ICL_EXAMPLE_IDS = {
    "nominal_1": "S074",   # nominal_single_aircraft  (7 entries, clean)
    "nominal_2": "S096",   # nominal_instrument_approach
    "warning_1": "S050",   # silent_traffic           (9 entries)
    "warning_2": "S036",   # missing_position_calls
    "hazard_1":  "S031",   # runway_incursion_risk    (8 entries)
    "hazard_2":  "S003",   # simultaneous_final
}


SYSTEM_PROMPT = """You are an automated aviation safety monitoring system for Half Moon Bay Airport (KHAF), a non-towered airport near San Francisco, California.

Your task is to analyze CTAF (Common Traffic Advisory Frequency) radio communications at KHAF and classify the safety status of the current traffic situation.

You will be given:
1. METAR weather data for KHAF
2. A CTAF radio transcript (SRT format with timestamps)

━━━ CLASSIFICATION RULES ━━━

Classify as exactly ONE of: nominal | warning | hazard

NOMINAL — All is well.
  • All required position calls are present (crosswind, downwind, base, final)
  • Traffic is sequenced and separated with no conflicts
  • Weather is VMC and appropriate for operations
  • Example: single aircraft announcing each leg, no other traffic

WARNING — A POTENTIAL problem exists but NO collision is imminent yet.
  Use warning when the situation is DEVELOPING but pilots still have time to self-correct:
  • An aircraft is flying the wrong pattern direction but has not yet conflicted
  • Traffic is observed (ADS-B / visual) but has not made radio calls (NORDO / silent traffic)
  • Two aircraft are converging on final but separation still exists (>0.5 NM)
  • An aircraft entered the pattern incorrectly (wrong entry point, wrong altitude)
  • A go-around was required due to spacing — but it was called and executed
  • Missing position calls from ONE aircraft with no immediate traffic conflict
  • IMC conditions approaching but VFR/IFR traffic not yet in conflict
  Key question: "Can the pilots resolve this themselves with standard advisory actions?"
  If YES → warning

HAZARD — A collision or serious incident is IMMINENT or ALREADY occurring.
  Use hazard ONLY when the situation requires IMMEDIATE evasive action right now:
  • Two aircraft are simultaneously on final for the same runway (< 0.5 NM separation)
  • An aircraft is on the runway while another is on short final (runway incursion)
  • A VFR aircraft is flying in IMC (instrument meteorological conditions) without a clearance
  • An aircraft has announced the wrong runway and is lined up on the wrong approach
  • A mid-air collision risk is present — aircraft are at the same altitude and converging
  Key question: "Would a CTAF advisory say 'IMMEDIATELY' or 'SAFETY ALERT'?"
  If YES → hazard

━━━ CRITICAL DISTINCTION ━━━
The difference between warning and hazard is IMMINENCE, not severity:
  • Pattern conflict with room to maneuver → WARNING
  • Two aircraft about to touch the same runway simultaneously → HAZARD
  • NORDO aircraft observed → WARNING (potential, not imminent)
  • VFR aircraft confirmed in IMC → HAZARD (emergency NOW)
  • Go-around executed, spacing restored → WARNING (was a conflict, now managed)
  • Wrong runway announced, still inbound → HAZARD (immediate correction required)

━━━ CTAF RULES (FAA AC 90-66C) ━━━
- Pilots MUST self-announce: crosswind, downwind, base, final, runway clear
- Straight-in: announce at 10, 5, and 3 NM
- Go-around MUST be announced immediately
- No ATC — pilots are solely responsible for separation

Respond with ONLY this JSON (no other text):
{
  "label": "<nominal|warning|hazard>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence: state the specific safety factor and why it is warning vs hazard>"
}"""


def format_scenario_input(scenario: dict) -> str:
    """Build the user-turn content for a single scenario."""
    metar = scenario["metar"]
    tx    = scenario["transcript_ground_truth"]

    metar_block = (
        f"METAR: {metar['raw']}\n"
        f"Conditions: {metar['description']}"
    )
    if metar.get("ceiling_ft"):
        metar_block += f" | Ceiling: {metar['ceiling_ft']} ft"
    metar_block += f" | Wind: {metar['wind_kt']} kt"

    tx_lines = "\n".join(
        f"[{e['timestamp']}] {e['text']}"
        for e in tx
    )

    return (
        f"--- WEATHER ---\n{metar_block}\n\n"
        f"--- CTAF TRANSCRIPT ---\n{tx_lines}\n\n"
        f"Classify the safety level:"
    )


def format_icl_example(scenario: dict) -> tuple[str, str]:
    """Return (user_content, assistant_content) for a standard ICL example."""
    user = format_scenario_input(scenario)
    assistant = json.dumps({
        "label":      scenario["label"],
        "confidence": 0.95,
        "reasoning":  scenario["ground_truth_advisory"][:120].rstrip(".") + ".",
    }, indent=None)
    return user, assistant


def format_icl_example_cot(scenario: dict) -> tuple[str, str]:
    """Return (user_content, assistant_content) for a CoT ICL example."""
    user = format_scenario_input(scenario) + COT_THINK_INSTRUCTION

    # Build a canned step-by-step reasoning from the ground truth fields
    label    = scenario["label"]
    advisory = scenario["ground_truth_advisory"][:180].rstrip(".") + "."
    htype    = scenario["hazard_type"].replace("_", " ")

    if label == "nominal":
        reasoning_steps = (
            f"1. Aircraft are present and making required position calls.\n"
            f"2. No communication gaps detected.\n"
            f"3. No conflict — all traffic properly sequenced.\n"
            f"4. No SAFETY ALERT or IMMEDIATELY language needed.\n"
            f"→ This is a nominal situation."
        )
    elif label == "warning":
        reasoning_steps = (
            f"1. Identified scenario type: {htype}.\n"
            f"2. A communication gap or potential conflict is present.\n"
            f"3. The conflict is POTENTIAL — pilots still have time to self-correct.\n"
            f"4. A CTAF advisory would recommend caution, not immediate evasion.\n"
            f"→ This is a warning situation."
        )
    elif label == "hazard":
        reasoning_steps = (
            f"1. Identified scenario type: {htype}.\n"
            f"2. An active conflict or emergency condition is present.\n"
            f"3. The situation is IMMINENT — immediate evasive action is required NOW.\n"
            f"4. A CTAF advisory would use 'SAFETY ALERT' or 'IMMEDIATELY'.\n"
            f"→ This is a hazard situation."
        )
    else:  # binary "danger" — used when --binary collapses warning + hazard
        reasoning_steps = (
            f"1. Identified scenario type: {htype}.\n"
            f"2. A communication gap, conflict, or unsafe condition is present.\n"
            f"3. A CTAF advisory would flag this for caution, alert, or emergency.\n"
            f"→ This is a danger situation."
        )

    final_json = json.dumps({
        "label":      label,
        "confidence": 0.95,
        "reasoning":  advisory,
    }, indent=None)

    assistant = f"{reasoning_steps}\n\n{final_json}"
    return user, assistant


def build_messages(
    scenario: dict,
    icl_examples: list[dict],
    strategy: str,
    model_key: str = "qwen",
    cot: bool = False,
) -> list[dict]:
    """Build the messages list for a given strategy."""
    NO_SYSTEM_ROLE = {"gemma"}
    use_system_role = model_key not in NO_SYSTEM_ROLE
    asst_role = "model" if model_key == "gemma" else "assistant"

    msgs = []
    if use_system_role:
        msgs.append({"role": "system", "content": SYSTEM_PROMPT})

    # ICL examples
    if strategy in ("one_shot", "few_shot"):
        n_per_label = 1 if strategy == "one_shot" else 2
        for label in LABELS:
            label_exs = [e for e in icl_examples if e["label"] == label]
            for ex in label_exs[:n_per_label]:
                u, a = (format_icl_example_cot(ex) if cot
                        else format_icl_example(ex))
                msgs.append({"role": "user",      "content": u})
                msgs.append({"role": asst_role,   "content": a})

    # Final scenario turn
    scenario_content = format_scenario_input(scenario)
    if cot:
        scenario_content = scenario_content + COT_THINK_INSTRUCTION

    if use_system_role:
        msgs.append({"role": "user", "content": scenario_content})
    else:
        # Gemma: prepend system prompt to the first user turn
        first_user_content = f"{SYSTEM_PROMPT}\n\n{scenario_content}"
        if msgs:
            first_user_idx = next(
                (i for i, m in enumerate(msgs) if m["role"] == "user"), None
            )
            if first_user_idx is not None:
                msgs[first_user_idx]["content"] = (
                    f"{SYSTEM_PROMPT}\n\n{msgs[first_user_idx]['content']}"
                )
                msgs.append({"role": "user", "content": scenario_content})
            else:
                msgs.append({"role": "user", "content": first_user_content})
        else:
            msgs.append({"role": "user", "content": first_user_content})

    return msgs


_LABEL_SYNONYMS = {
    "safe":     "nominal",
    "normal":   "nominal",
    "clear":    "nominal",
    "ok":       "nominal",
    "caution":  "warning",
    "warn":     "warning",
    "alert":    "warning",
    "danger":   "hazard",
    "critical": "hazard",
    "emergency":"hazard",
    "unsafe":   "hazard",
}

def parse_response(raw: str) -> dict:
    """Parse the model output into {label, confidence, reasoning, parse_ok}."""
    # Strip markdown fences
    clean = re.sub(r"```[a-z]*\n?", "", raw).strip()

    # Try JSON parse
    try:
        # Find first {...} block
        match = re.search(r"\{.*?\}", clean, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            label = str(obj.get("label", "")).strip().lower()
            label = _LABEL_SYNONYMS.get(label, label)
            if label not in LABELS:
                # Fallback: scan full text for label
                label = _extract_label_from_text(clean)
            return {
                "label":      label,
                "confidence": float(obj.get("confidence", 0.5)),
                "reasoning":  str(obj.get("reasoning", ""))[:200],
                "parse_ok":   label in LABELS,
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: keyword search in raw text
    label = _extract_label_from_text(clean)
    return {
        "label":      label,
        "confidence": 0.5,
        "reasoning":  clean[:200],
        "parse_ok":   label in LABELS,
    }


def _extract_label_from_text(text: str) -> str:
    tl = text.lower()
    # Check in priority order: most-severe first when present
    priority = ["hazard", "warning", "danger", "nominal"]
    for lbl in priority:
        if lbl in LABELS and lbl in tl:
            return lbl
    for syn, lbl in _LABEL_SYNONYMS.items():
        if syn in tl and lbl in LABELS:
            return lbl
    return "unknown"


def load_model(model_key: str, quantize: str, log: logging.Logger):
    """Load HuggingFace model + tokenizer. Returns (model, tokenizer, device)."""
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                               BitsAndBytesConfig)

    cfg  = MODELS[model_key]
    hfid = cfg["hf_id"]
    log.info(f"Loading {cfg['display']} ({hfid})")
    log.info(f"  Quantization: {quantize or 'none (fp16)'}")

    tokenizer = AutoTokenizer.from_pretrained(
        hfid,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN", None),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    dtype = torch.float16

    if quantize == "4bit":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        dtype = None
    elif quantize == "8bit":
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        dtype = None

    kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
    )
    if quant_cfg:
        kwargs["quantization_config"] = quant_cfg
    else:
        kwargs["dtype"] = dtype

    if model_key == "gemma":
        hf_token = os.environ.get("HF_TOKEN", None)
        if hf_token:
            kwargs["token"] = hf_token

    model = AutoModelForCausalLM.from_pretrained(hfid, **kwargs)
    model.eval()

    device = next(model.parameters()).device
    log.info(f"  Loaded on {device}")
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    messages: list[dict],
    model_cfg: dict,
) -> tuple[str, float]:
    """Run a single inference call. Returns (raw_output_text, latency_seconds)."""
    import torch

    t0 = time.time()

    # apply_chat_template may return a raw tensor or a BatchEncoding dict
    # depending on the transformers version and tokenizer implementation.
    # We handle both cases explicitly.
    chat_out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # BatchEncoding (dict-like) → extract input_ids tensor
    if hasattr(chat_out, "input_ids"):
        input_ids      = chat_out.input_ids.to(model.device)
        attention_mask = chat_out.attention_mask.to(model.device) \
                         if hasattr(chat_out, "attention_mask") else None
    elif isinstance(chat_out, dict):
        input_ids      = chat_out["input_ids"].to(model.device)
        attention_mask = chat_out.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
    else:
        # Raw tensor (standard behaviour)
        input_ids      = chat_out.to(model.device)
        attention_mask = None

    prompt_len = input_ids.shape[-1]

    gen_kwargs = dict(
        input_ids      = input_ids,
        max_new_tokens = model_cfg["max_new_tokens"],
        do_sample      = model_cfg["temperature"] > 0,
        pad_token_id   = tokenizer.pad_token_id,
        eos_token_id   = tokenizer.eos_token_id,
    )
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    if model_cfg["temperature"] > 0:
        gen_kwargs["temperature"] = model_cfg["temperature"]

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    # Decode only the newly generated tokens (strip the prompt)
    new_tokens = output_ids[0][prompt_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    latency = time.time() - t0
    return raw, latency


def run_inference_cot(
    model,
    tokenizer,
    messages: list[dict],   # first-turn messages (already includes COT_THINK_INSTRUCTION)
    model_cfg: dict,
    asst_role: str = "assistant",
) -> tuple[str, str, float]:
    """Two-turn CoT inference for open-source models."""
    import torch
    t0 = time.time()

    cot_cfg = dict(model_cfg)
    cot_cfg["max_new_tokens"] = COT_MAX_NEW_TOKENS

    chat_out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if hasattr(chat_out, "input_ids"):
        input_ids      = chat_out.input_ids.to(model.device)
        attention_mask = getattr(chat_out, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
    elif isinstance(chat_out, dict):
        input_ids      = chat_out["input_ids"].to(model.device)
        attention_mask = chat_out.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
    else:
        input_ids = chat_out.to(model.device); attention_mask = None

    prompt_len = input_ids.shape[-1]
    gen_kwargs = dict(input_ids=input_ids, max_new_tokens=COT_MAX_NEW_TOKENS,
                      do_sample=False, pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id)
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)
    cot_text = tokenizer.decode(
        output_ids[0][prompt_len:], skip_special_tokens=True).strip()

    messages_t2 = messages + [
        {"role": asst_role, "content": cot_text},
        {"role": "user",    "content": COT_EXTRACT_INSTRUCTION},
    ]
    chat_out2 = tokenizer.apply_chat_template(
        messages_t2, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if hasattr(chat_out2, "input_ids"):
        ids2  = chat_out2.input_ids.to(model.device)
        mask2 = getattr(chat_out2, "attention_mask", None)
        if mask2 is not None: mask2 = mask2.to(model.device)
    elif isinstance(chat_out2, dict):
        ids2  = chat_out2["input_ids"].to(model.device)
        mask2 = chat_out2.get("attention_mask", None)
        if mask2 is not None: mask2 = mask2.to(model.device)
    else:
        ids2 = chat_out2.to(model.device); mask2 = None

    pl2 = ids2.shape[-1]
    gkw2 = dict(input_ids=ids2, max_new_tokens=150, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
    if mask2 is not None:
        gkw2["attention_mask"] = mask2
    if model_cfg.get("use_cache") is False:
        gkw2["use_cache"] = False

    with torch.no_grad():
        out2 = model.generate(**gkw2)
    json_text = tokenizer.decode(out2[0][pl2:], skip_special_tokens=True).strip()

    return cot_text, json_text, time.time() - t0


def score_classes_hf(model, tokenizer, messages, classes) -> dict:
    """Append a constrained-final-answer prompt and read next-token logits for each"""
    import torch
    import math

    msgs = list(messages) + [{
        "role":    "user",
        "content": ("Output exactly one word and nothing else, chosen from this "
                    "list: " + ", ".join(classes) + "."),
    }]
    chat_out = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if hasattr(chat_out, "input_ids"):
        ids = chat_out.input_ids.to(model.device)
    elif isinstance(chat_out, dict):
        ids = chat_out["input_ids"].to(model.device)
    else:
        ids = chat_out.to(model.device)

    with torch.no_grad():
        logits = model(ids).logits[0, -1]

    cand = []
    for c in classes:
        # Tokenizers often prefix word tokens with a leading space; try a few
        # spellings and keep the highest-logit match.
        best = -1e9
        for v in (" " + c, c, c.capitalize(), " " + c.capitalize()):
            tok_ids = tokenizer.encode(v, add_special_tokens=False)
            if tok_ids:
                lg = logits[tok_ids[0]].item()
                if lg > best:
                    best = lg
        cand.append(best)

    mx = max(cand)
    exps = [math.exp(l - mx) for l in cand]
    total = sum(exps) or 1.0
    return {c: e / total for c, e in zip(classes, exps)}


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute per-class and macro precision, recall, F1 + accuracy."""
    labels = LABELS
    K = len(labels)
    n = len(y_true)

    # Confusion matrix  [true × pred]
    cm = np.zeros((K, K), dtype=int)
    label2i = {l: i for i, l in enumerate(labels)}
    for yt, yp in zip(y_true, y_pred):
        ti = label2i.get(yt, -1)
        pi = label2i.get(yp, -1)
        if ti >= 0 and pi >= 0:
            cm[ti, pi] += 1

    per_class = {}
    for i, lbl in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[lbl] = {"precision": prec, "recall": rec, "f1": f1,
                           "support": int(cm[i, :].sum())}

    macro_p  = np.mean([per_class[l]["precision"] for l in labels])
    macro_r  = np.mean([per_class[l]["recall"]    for l in labels])
    macro_f1 = np.mean([per_class[l]["f1"]        for l in labels])
    accuracy = cm.diagonal().sum() / n if n > 0 else 0.0
    parse_ok_rate = sum(1 for p in y_pred if p in LABELS) / n if n > 0 else 0.0

    return {
        "accuracy":     round(accuracy,     4),
        "macro_p":      round(macro_p,       4),
        "macro_r":      round(macro_r,       4),
        "macro_f1":     round(macro_f1,      4),
        "parse_ok":     round(parse_ok_rate, 4),
        "per_class":    {l: {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in per_class[l].items()}
                         for l in labels},
        "confusion_matrix": cm.tolist(),
        "n": n,
    }


def compute_per_hazard_metrics(records: list) -> dict:
    """F1 per hazard_type for a given model+strategy run."""
    by_ht = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for r in records:
        ht = r["hazard_type"]
        by_ht[ht]["y_true"].append(r["ground_truth"])
        by_ht[ht]["y_pred"].append(r["predicted"])

    out = {}
    for ht, d in by_ht.items():
        m = compute_metrics(d["y_true"], d["y_pred"])
        out[ht] = {
            "accuracy":  m["accuracy"],
            "macro_f1":  m["macro_f1"],
            "n":         m["n"],
        }
    return out


# Paper-quality style
STYLE = {
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        180,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
}

LABEL_COLORS = {"nominal": "#639922", "warning": "#BA7517", "hazard": "#A32D2D",
                "danger":  "#A32D2D"}
MODEL_COLORS  = {"Qwen2.5-7B": "#378ADD", "Mistral-7B": "#7F77DD", "Gemma-2-9B": "#1D9E75"}
STRAT_MARKERS = {"zero_shot": "o", "one_shot": "s", "few_shot": "^"}
STRAT_LABELS  = {"zero_shot": "Zero-shot", "one_shot": "One-shot", "few_shot": "Few-shot"}


def plot_confusion_matrix(cm_data: list, model: str, strategy: str, out_path: Path):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        cm = np.array(cm_data)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        K = len(LABELS)
        ticks = range(K)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([l.capitalize() for l in LABELS])
        ax.set_yticklabels([l.capitalize() for l in LABELS])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{model} — {STRAT_LABELS[strategy]}")

        for i in range(K):
            for j in range(K):
                val    = cm[i, j]
                norm_v = cm_norm[i, j]
                color  = "white" if norm_v > 0.55 else "black"
                ax.text(j, i, f"{val}\n({norm_v:.0%})",
                        ha="center", va="center", fontsize=9.5,
                        color=color, fontweight="bold" if i == j else "normal")

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def plot_f1_by_model_strategy(all_metrics: dict, out_path: Path):
    """Grouped bar chart: model × strategy, metric = macro F1."""
    models     = list(MODELS.keys())
    strategies = STRATEGIES
    model_names = [MODELS[m]["short_name"] for m in models]

    x     = np.arange(len(models))
    width = 0.25
    strat_colors = ["#B5D4F4", "#378ADD", "#0C447C"]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for si, strat in enumerate(strategies):
            vals = []
            for mk in models:
                key = f"{mk}_{strat}"
                vals.append(all_metrics.get(key, {}).get("macro_f1", 0.0))
            bars = ax.bar(x + (si - 1) * width, vals, width,
                          label=STRAT_LABELS[strat],
                          color=strat_colors[si],
                          edgecolor="white", linewidth=0.8)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.008,
                            f"{val:.2f}", ha="center", va="bottom",
                            fontsize=8.5, color="#2C2C2A")

        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel("Macro F1 score")
        ax.set_ylim(0, 1.08)
        ax.set_title("Macro F1 by model and prompting strategy")
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def plot_per_class_f1(all_metrics: dict, out_path: Path):
    """Heatmap: rows = model+strategy, cols = per-class F1."""
    rows = []
    per_label_f1 = {l: [] for l in LABELS}
    for mk in MODELS:
        for st in STRATEGIES:
            key = f"{mk}_{st}"
            if key not in all_metrics:
                continue
            short = MODELS[mk]['short_name']
            strat_label = STRAT_LABELS[st]
            rows.append(f"{short}\n{strat_label}")
            pc = all_metrics[key]["per_class"]
            for l in LABELS:
                per_label_f1[l].append(pc[l]["f1"])

    data = np.array([per_label_f1[l] for l in LABELS])   # shape (K, N)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.9), 3.8))
        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1,
                       aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                     label="F1 score")

        K = len(LABELS)
        ax.set_yticks(range(K))
        ax.set_yticklabels([l.capitalize() for l in LABELS])
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(rows, fontsize=9)
        ax.set_title("Per-class F1 heatmap across models and prompting strategies")

        for i in range(K):
            for j in range(len(rows)):
                v = data[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if v < 0.35 or v > 0.80 else "black")

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def plot_strategy_improvement(all_metrics: dict, out_path: Path):
    """Line plot: macro F1 vs prompting strategy per model."""
    strat_x = [0, 1, 2]
    strat_labels = [STRAT_LABELS[s] for s in STRATEGIES]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for mk, mcfg in MODELS.items():
            vals = []
            for st in STRATEGIES:
                key = f"{mk}_{st}"
                vals.append(all_metrics.get(key, {}).get("macro_f1", None))
            # Only plot if we have results
            if any(v is not None for v in vals):
                y = [v if v is not None else float("nan") for v in vals]
                color = MODEL_COLORS[mcfg["short_name"]]
                ax.plot(strat_x, y, marker="o", linewidth=2,
                        markersize=8, color=color, label=mcfg["display"])
                for xi, yi in zip(strat_x, y):
                    if not np.isnan(yi):
                        ax.annotate(f"{yi:.2f}", (xi, yi),
                                    textcoords="offset points",
                                    xytext=(0, 8), ha="center",
                                    fontsize=8.5, color=color)

        ax.set_xticks(strat_x)
        ax.set_xticklabels(strat_labels)
        ax.set_ylabel("Macro F1 score")
        ax.set_ylim(0, 1.05)
        ax.set_title("Effect of in-context learning on classification performance")
        ax.legend(loc="lower right", frameon=True)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def plot_per_hazard_f1(all_hazard_metrics: dict, out_path: Path):
    """Horizontal bar chart: per hazard-type accuracy for the best"""
    # Find best combo by macro_f1
    best_key = max(
        (k for k in all_hazard_metrics),
        key=lambda k: np.mean([v["macro_f1"]
                                for v in all_hazard_metrics[k].values()])
    )
    ht_data = all_hazard_metrics[best_key]

    hazard_types = sorted(ht_data.keys())
    accuracies   = [ht_data[ht]["accuracy"] for ht in hazard_types]
    counts       = [ht_data[ht]["n"] for ht in hazard_types]

    label_for_ht = {
        "simultaneous_final":        "hazard",
        "wrong_runway_announcement": "hazard",
        "imc_vfr_conflict":          "hazard",
        "runway_incursion_risk":      "hazard",
        "missing_position_calls":    "warning",
        "pattern_conflict":          "warning",
        "silent_traffic":            "warning",
        "go_around_conflict":        "warning",
        "improper_entry":            "warning",
        "nominal_single_aircraft":       "nominal",
        "nominal_multi_aircraft":        "nominal",
        "nominal_instrument_approach":   "nominal",
    }
    # In binary mode, collapse hazard_type -> {nominal, danger} so colors
    # match the LABELS legend.
    if "danger" in LABELS:
        label_for_ht = {ht: ("nominal" if v == "nominal" else "danger")
                         for ht, v in label_for_ht.items()}
    colors = [LABEL_COLORS[label_for_ht.get(ht, "nominal")] for ht in hazard_types]
    labels_clean = [ht.replace("_", " ").capitalize() for ht in hazard_types]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(labels_clean, accuracies, color=colors,
                       edgecolor="white", linewidth=0.5, height=0.65)
        for bar, acc, cnt in zip(bars, accuracies, counts):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{acc:.2f}  (n={cnt})", va="center", fontsize=8.5)

        ax.set_xlim(0, 1.18)
        ax.set_xlabel("Accuracy")
        best_key_clean = best_key.replace('_', ' ')
        ax.set_title(f"Per-hazard-type accuracy\n(best model: {best_key_clean})")

        patches = [mpatches.Patch(color=LABEL_COLORS[l], label=l.capitalize())
                   for l in LABELS]
        ax.legend(handles=patches, loc="lower right", frameon=True)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def plot_latency(all_metrics: dict, out_path: Path):
    """Box-style bar chart: average inference latency per model."""
    model_keys   = list(MODELS.keys())
    model_names  = [MODELS[m]["short_name"] for m in model_keys]
    avg_latency  = []

    for mk in model_keys:
        lats = []
        for st in STRATEGIES:
            key = f"{mk}_{st}"
            if key in all_metrics and "avg_latency_s" in all_metrics[key]:
                lats.append(all_metrics[key]["avg_latency_s"])
        avg_latency.append(np.mean(lats) if lats else 0.0)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        colors = [MODEL_COLORS[MODELS[m]["short_name"]] for m in model_keys]
        bars = ax.bar(model_names, avg_latency, color=colors,
                      edgecolor="white", width=0.5)
        for bar, val in zip(bars, avg_latency):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        f"{val:.2f}s", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Avg. inference latency (s/scenario)")
        ax.set_title("Inference latency by model (RTX 5090)")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def make_main_results_table(all_metrics: dict) -> str:
    """Paper Table 1: Model × Strategy with Acc / P / R / F1 per class + macro."""
    K = len(LABELS)
    # Two leading cols (Model, Strategy) + 3 cols (P, R, F1) per class.
    col_spec = "ll" + "ccc" * K
    # Per-class header groups
    header_top = " & ".join(
        [r"", r""] +
        [r"\multicolumn{3}{c}{\textbf{" + l.capitalize() + r"}}" for l in LABELS]
    ) + r" \\"
    cmidrules = "".join(
        rf"\cmidrule(lr){{{2 + 3 * i + 1}-{2 + 3 * i + 3}}}"
        for i in range(K)
    )
    header_low = (
        r"\textbf{Model} & \textbf{Strategy}"
        + " & P & R & F1" * K
        + r" \\"
    )

    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        rf"\caption{{Classification results across open-source models and prompting strategies on the CTAF-KHAF-Synthetic dataset ({K}-class task: " + " / ".join(l.capitalize() for l in LABELS) + r"). Best macro F1 per model is \textbf{bold}.}",
        r"\label{tab:main_results}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_top,
        cmidrules,
        header_low,
        r"\midrule",
    ]

    for mk, mcfg in MODELS.items():
        best_f1 = max(
            (all_metrics.get(f"{mk}_{st}", {}).get("macro_f1", 0) for st in STRATEGIES),
            default=0,
        )
        rows_for_model = []
        for si, st in enumerate(STRATEGIES):
            key = f"{mk}_{st}"
            m   = all_metrics.get(key, {})
            if not m:
                continue
            pc   = m["per_class"]
            mf1  = m["macro_f1"]
            bold = mf1 == best_f1

            def fmt(v):
                s = f"{v:.2f}"
                return rf"\textbf{{{s}}}" if bold else s

            multirow = (rf"\multirow{{{len(STRATEGIES)}}}{{*}}{{" + mcfg['display'] + "}"
                        if si == 0 else "")
            cells = []
            for l in LABELS:
                cells.extend([
                    fmt(pc[l]["precision"]),
                    fmt(pc[l]["recall"]),
                    fmt(pc[l]["f1"]),
                ])
            row = (
                f"  {multirow}"
                f" & {STRAT_LABELS[st]} & "
                + " & ".join(cells)
                + r" \\"
            )
            rows_for_model.append(row)
        lines.extend(rows_for_model)
        lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def make_summary_table(all_metrics: dict) -> str:
    """Table 2: compact accuracy + macro F1 + latency summary."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Summary: accuracy, macro F1, and inference latency per model and prompting strategy.}",
        r"\label{tab:summary}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Strategy} & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{Latency (s)} \\",
        r"\midrule",
    ]
    for mk, mcfg in MODELS.items():
        for si, st in enumerate(STRATEGIES):
            key = f"{mk}_{st}"
            m   = all_metrics.get(key, {})
            if not m:
                continue
            lat = m.get("avg_latency_s", 0)
            multirow = r"\multirow{3}{*}{" + mcfg['short_name'] + "}" if si == 0 else ""
            newline  = r"\\"
            row = (
                f"  {multirow}"
                f" & {STRAT_LABELS[st]}"
                f" & {m['accuracy']:.3f}"
                f" & {m['macro_f1']:.3f}"
                f" & {lat:.2f} {newline}"
            )
            lines.append(row)
        lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def run_experiments(args, log: logging.Logger, out_dir: Path):
    import torch

    log.info(f"Loading dataset: {args.dataset}")
    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
    all_scenarios = data["scenarios"]

    if getattr(args, "binary", False):
        for s in all_scenarios:
            s["label"] = collapse_label(s["label"])
        log.info("  Binary mode: labels collapsed -> nominal vs danger.")

    # Pull ICL examples out of test set
    icl_ids = set(ICL_EXAMPLE_IDS.values())
    icl_pool = {s["scenario_id"]: s for s in all_scenarios if s["scenario_id"] in icl_ids}
    test_set = [s for s in all_scenarios if s["scenario_id"] not in icl_ids]

    # Ordered ICL examples: nominal×2, warning×2, hazard×2
    icl_ordered = [
        icl_pool[ICL_EXAMPLE_IDS["nominal_1"]],
        icl_pool[ICL_EXAMPLE_IDS["nominal_2"]],
        icl_pool[ICL_EXAMPLE_IDS["warning_1"]],
        icl_pool[ICL_EXAMPLE_IDS["warning_2"]],
        icl_pool[ICL_EXAMPLE_IDS["hazard_1"]],
        icl_pool[ICL_EXAMPLE_IDS["hazard_2"]],
    ]

    log.info(f"Test set: {len(test_set)} scenarios  |  ICL pool: {len(icl_pool)} scenarios")
    log.info(f"Models:     {args.models}")
    log.info(f"Strategies: {args.strategies}")
    log.info(f"Quantize:   {args.quantize or 'none'}")

    all_metrics      = {}
    all_hazard_mets  = {}

    use_cot = getattr(args, "cot", False)
    force   = getattr(args, "force", False)
    cot_suffix = "_cot" if use_cot else ""
    asst_role_map = {"gemma": "model", "qwen": "assistant",
                     "mistral": "assistant"}

    for model_key in ([] if getattr(args, "closed_source_only", False) else args.models):
        mcfg = MODELS[model_key]
        log.info("=" * 60)
        log.info(f"Model: {mcfg['display']}  (CoT={'ON' if use_cot else 'OFF'})")

        model, tokenizer = load_model(model_key, args.quantize, log)
        asst_role = asst_role_map.get(model_key, "assistant")

        for strategy in args.strategies:
            run_key  = f"{model_key}_{strategy}{cot_suffix}"
            raw_path = out_dir / "raw" / f"{run_key}.json"

            if raw_path.exists() and not force:
                log.info(f"  Strategy: {STRAT_LABELS[strategy]}{cot_suffix}  "
                         f"[RESUMING from {raw_path.name}]")
                with open(raw_path) as f:
                    saved = json.load(f)
                raw_records = saved["records"]
                y_true = [r["ground_truth"] for r in raw_records]
                y_pred = [r["predicted"]    for r in raw_records]
                latencies = [r["latency_s"] for r in raw_records]
                metrics = compute_metrics(y_true, y_pred)
                metrics["avg_latency_s"] = round(float(np.mean(latencies)), 3)
                metrics.update({"model": mcfg["display"], "model_key": model_key,
                                  "strategy": strategy, "run_key": run_key,
                                  "cot": use_cot})
                all_metrics[run_key]     = metrics
                all_hazard_mets[run_key] = compute_per_hazard_metrics(raw_records)
                (out_dir / "figures").mkdir(parents=True, exist_ok=True)
                plot_confusion_matrix(metrics["confusion_matrix"],
                                      mcfg["short_name"], strategy,
                                      out_dir / "figures" / f"cm_{run_key}.pdf")
                log.info(f"    → Accuracy={metrics['accuracy']:.3f}  "
                         f"MacroF1={metrics['macro_f1']:.3f}")
                continue

            log.info(f"  Strategy: {STRAT_LABELS[strategy]}{cot_suffix}")
            raw_records, y_true, y_pred, latencies = [], [], [], []

            for idx, scenario in enumerate(test_set):
                sid      = scenario["scenario_id"]
                messages = build_messages(scenario, icl_ordered, strategy,
                                          model_key=model_key, cot=use_cot)
                gt       = scenario["label"]

                if use_cot:
                    cot_text, json_out, latency = run_inference_cot(
                        model, tokenizer, messages, mcfg, asst_role=asst_role)
                    parsed = parse_response(json_out)
                    raw_out = json_out
                    cot_stored = cot_text
                else:
                    raw_out, latency = run_inference(model, tokenizer, messages, mcfg)
                    parsed = parse_response(raw_out)
                    cot_stored = None

                try:
                    cls_scores = score_classes_hf(
                        model, tokenizer, messages, LABELS)
                except Exception:
                    cls_scores = {}
                score_source = "logprobs"
                if not cls_scores:
                    conf = parsed["confidence"]
                    if parsed["label"] in LABELS:
                        cls_scores = {l: ((1.0 - conf) / max(len(LABELS) - 1, 1)
                                          if l != parsed["label"] else conf)
                                       for l in LABELS}
                    else:
                        cls_scores = {l: 1.0 / len(LABELS) for l in LABELS}
                    score_source = "confidence_fallback"

                rec = {
                    "scenario_id":   sid,
                    "hazard_type":   scenario["hazard_type"],
                    "ground_truth":  gt,
                    "predicted":     parsed["label"],
                    "score_source":  score_source,
                    "confidence":    parsed["confidence"],
                    "class_scores":  cls_scores,
                    "reasoning":     parsed["reasoning"],
                    "parse_ok":      parsed["parse_ok"],
                    "raw_output":    raw_out,
                    "cot_reasoning": cot_stored,
                    "latency_s":     round(latency, 3),
                    "correct":       gt == parsed["label"],
                    "model":         mcfg["short_name"],
                    "strategy":      strategy,
                    "cot":           use_cot,
                    "metar_ceiling": scenario["metar"].get("ceiling_ft"),
                    "metar_vis":     scenario["metar"]["visibility_sm"],
                }
                raw_records.append(rec)
                y_true.append(gt); y_pred.append(parsed["label"])
                latencies.append(latency)
                status = "✓" if rec["correct"] else "✗"
                log.info(f"    [{idx+1:3d}/{len(test_set)}] {sid} "
                         f"GT={gt:7s} PRED={parsed['label']:7s} {status} "
                         f"({latency:.2f}s)")

            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_path, "w") as f:
                json.dump({"run_key": run_key, "model": mcfg["display"],
                           "strategy": strategy, "cot": use_cot,
                           "timestamp": datetime.now().isoformat(),
                           "n_test": len(test_set), "icl_ids": list(icl_ids),
                           "records": raw_records}, f, indent=2)

            metrics = compute_metrics(y_true, y_pred)
            metrics["avg_latency_s"] = round(np.mean(latencies), 3)
            metrics["model"]         = mcfg["display"]
            metrics["model_key"]     = model_key
            metrics["strategy"]      = strategy
            metrics["run_key"]       = run_key
            all_metrics[run_key]     = metrics

            haz_metrics = compute_per_hazard_metrics(raw_records)
            all_hazard_mets[run_key] = haz_metrics

            log.info(
                f"    → Accuracy={metrics['accuracy']:.3f}  "
                f"MacroF1={metrics['macro_f1']:.3f}  "
                f"ParseOK={metrics['parse_ok']:.3f}"
            )

            (out_dir / "figures").mkdir(parents=True, exist_ok=True)
            cm_path = out_dir / "figures" / f"cm_{run_key}.pdf"
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                mcfg["short_name"], strategy, cm_path
            )
            log.info(f"    Saved confusion matrix → {cm_path.name}")

        del model, tokenizer
        torch.cuda.empty_cache()
        log.info(f"  GPU memory freed after {mcfg['display']}")

    if getattr(args, "closed_source_model", None):
        cs_key  = args.closed_source_model
        cs_cfg  = OPENAI_MODELS[cs_key]
        client  = load_openai_client(getattr(args, "openai_api_key", None))

        log.info("=" * 60)
        log.info(f"Closed-source model: {cs_cfg['display']}  "
                 f"(CoT={'ON' if use_cot else 'OFF'})")
        log.info(f"  Est. cost: {openai_cost_estimate(len(test_set), cs_key, len(args.strategies), cot=use_cot)}")

        for strategy in args.strategies:
            run_key  = f"{cs_key}_{strategy}{cot_suffix}"
            raw_path = out_dir / "raw" / f"{run_key}.json"

            # Resume
            if raw_path.exists() and not force:
                log.info(f"  {STRAT_LABELS[strategy]}{cot_suffix} "
                         f"[RESUMING from {raw_path.name}]")
                with open(raw_path) as f:
                    saved = json.load(f)
                raw_records = saved["records"]
                y_true      = [r["ground_truth"] for r in raw_records]
                y_pred      = [r["predicted"]    for r in raw_records]
                latencies   = [r["latency_s"]    for r in raw_records]
                metrics     = compute_metrics(y_true, y_pred)
                metrics["avg_latency_s"] = round(float(np.mean(latencies)), 3)
                metrics.update({"model": cs_cfg["display"], "model_key": cs_key,
                                  "strategy": strategy, "run_key": run_key,
                                  "cot": use_cot})
                all_metrics[run_key]     = metrics
                all_hazard_mets[run_key] = compute_per_hazard_metrics(raw_records)
                (out_dir / "figures").mkdir(parents=True, exist_ok=True)
                plot_confusion_matrix(metrics["confusion_matrix"],
                                      cs_cfg["short_name"], strategy,
                                      out_dir / "figures" / f"cm_{run_key}.pdf")
                log.info(f"    → Acc={metrics['accuracy']:.3f}  F1={metrics['macro_f1']:.3f}")
                continue

            log.info(f"  Strategy: {STRAT_LABELS[strategy]}{cot_suffix}")
            raw_records, y_true, y_pred, latencies = [], [], [], []

            for idx, scenario in enumerate(test_set):
                sid      = scenario["scenario_id"]
                messages = build_messages(scenario, icl_ordered, strategy,
                                          model_key="qwen", cot=use_cot)
                gt       = scenario["label"]

                if use_cot:
                    cot_text, json_out, latency = run_openai_inference_cot(
                        client, messages, cs_key, COT_EXTRACT_INSTRUCTION)
                    parsed    = parse_response(json_out)
                    raw_out   = json_out
                    cot_stored = cot_text
                else:
                    raw_out, latency = run_openai_inference(client, messages, cs_key)
                    parsed    = parse_response(raw_out)
                    cot_stored = None

                try:
                    cls_scores = score_classes_openai(
                        client, messages, cs_key, LABELS)
                except Exception:
                    cls_scores = {}
                score_source = "logprobs"
                if not cls_scores:
                    conf = parsed["confidence"]
                    if parsed["label"] in LABELS:
                        cls_scores = {l: ((1.0 - conf) / max(len(LABELS) - 1, 1)
                                          if l != parsed["label"] else conf)
                                       for l in LABELS}
                    else:
                        cls_scores = {l: 1.0 / len(LABELS) for l in LABELS}
                    score_source = "confidence_fallback"

                rec = {
                    "scenario_id":   sid,
                    "hazard_type":   scenario["hazard_type"],
                    "ground_truth":  gt,
                    "predicted":     parsed["label"],
                    "confidence":    parsed["confidence"],
                    "class_scores":  cls_scores,
                    "score_source":  score_source,
                    "reasoning":     parsed["reasoning"],
                    "parse_ok":      parsed["parse_ok"],
                    "raw_output":    raw_out,
                    "cot_reasoning": cot_stored,
                    "latency_s":     round(latency, 3),
                    "correct":       gt == parsed["label"],
                    "model":         cs_cfg["short_name"],
                    "strategy":      strategy,
                    "cot":           use_cot,
                    "metar_ceiling": scenario["metar"].get("ceiling_ft"),
                    "metar_vis":     scenario["metar"]["visibility_sm"],
                }
                raw_records.append(rec)
                y_true.append(gt); y_pred.append(parsed["label"])
                latencies.append(latency)
                status = "✓" if rec["correct"] else "✗"
                log.info(f"    [{idx+1:3d}/{len(test_set)}] {sid} "
                         f"GT={gt:7s} PRED={parsed['label']:7s} {status} "
                         f"({latency:.2f}s)")

            # Save raw
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_path, "w") as f:
                json.dump({"run_key": run_key, "model": cs_cfg["display"],
                           "strategy": strategy, "cot": use_cot,
                           "timestamp": datetime.now().isoformat(),
                           "n_test": len(test_set),
                           "records": raw_records}, f, indent=2)

            metrics = compute_metrics(y_true, y_pred)
            metrics["avg_latency_s"] = round(np.mean(latencies), 3)
            metrics.update({"model": cs_cfg["display"], "model_key": cs_key,
                              "strategy": strategy, "run_key": run_key,
                              "cot": use_cot})
            all_metrics[run_key]     = metrics
            all_hazard_mets[run_key] = compute_per_hazard_metrics(raw_records)
            log.info(f"    → Acc={metrics['accuracy']:.3f}  "
                     f"F1={metrics['macro_f1']:.3f}  "
                     f"ParseOK={metrics['parse_ok']:.3f}")
            (out_dir / "figures").mkdir(parents=True, exist_ok=True)
            plot_confusion_matrix(metrics["confusion_matrix"],
                                  cs_cfg["short_name"], strategy,
                                  out_dir / "figures" / f"cm_{run_key}.pdf")

    if getattr(args, "claude_model", None):
        cl_key = args.claude_model
        cl_cfg = CLAUDE_MODELS[cl_key]
        client = load_claude_client(getattr(args, "anthropic_api_key", None))

        log.info("=" * 60)
        log.info(f"Claude model: {cl_cfg['display']}  "
                 f"(CoT={'ON' if use_cot else 'OFF'})")
        log.info(f"  Est. cost: {claude_cost_estimate(len(test_set), cl_key, len(args.strategies), cot=use_cot)}")
        log.warning("  Note: Anthropic API does not expose token logprobs. "
                    "Per-class scores fall back to self-reported confidence.")

        for strategy in args.strategies:
            run_key  = f"{cl_key}_{strategy}{cot_suffix}"
            raw_path = out_dir / "raw" / f"{run_key}.json"

            if raw_path.exists() and not force:
                log.info(f"  {STRAT_LABELS[strategy]}{cot_suffix} "
                         f"[RESUMING from {raw_path.name}]")
                with open(raw_path) as f:
                    saved = json.load(f)
                raw_records = saved["records"]
                y_true = [r["ground_truth"] for r in raw_records]
                y_pred = [r["predicted"]    for r in raw_records]
                latencies = [r["latency_s"] for r in raw_records]
                metrics = compute_metrics(y_true, y_pred)
                metrics["avg_latency_s"] = round(float(np.mean(latencies)), 3)
                metrics.update({"model": cl_cfg["display"], "model_key": cl_key,
                                "strategy": strategy, "run_key": run_key,
                                "cot": use_cot})
                all_metrics[run_key]     = metrics
                all_hazard_mets[run_key] = compute_per_hazard_metrics(raw_records)
                (out_dir / "figures").mkdir(parents=True, exist_ok=True)
                plot_confusion_matrix(metrics["confusion_matrix"],
                                      cl_cfg["short_name"], strategy,
                                      out_dir / "figures" / f"cm_{run_key}.pdf")
                log.info(f"    → Acc={metrics['accuracy']:.3f}  F1={metrics['macro_f1']:.3f}")
                continue

            log.info(f"  Strategy: {STRAT_LABELS[strategy]}{cot_suffix}")
            raw_records, y_true, y_pred, latencies = [], [], [], []

            for idx, scenario in enumerate(test_set):
                sid      = scenario["scenario_id"]
                messages = build_messages(scenario, icl_ordered, strategy,
                                           model_key="qwen", cot=use_cot)
                gt       = scenario["label"]

                if use_cot:
                    cot_text, json_out, latency = run_claude_inference_cot(
                        client, messages, cl_key, COT_EXTRACT_INSTRUCTION)
                    parsed = parse_response(json_out)
                    raw_out = json_out
                    cot_stored = cot_text
                else:
                    raw_out, latency = run_claude_inference(client, messages, cl_key)
                    parsed = parse_response(raw_out)
                    cot_stored = None

                # Claude has no logprobs API — use confidence as a proxy score.
                conf = parsed["confidence"]
                if parsed["label"] in LABELS:
                    cls_scores = {l: ((1.0 - conf) / max(len(LABELS) - 1, 1)
                                      if l != parsed["label"] else conf)
                                   for l in LABELS}
                else:
                    cls_scores = {l: 1.0 / len(LABELS) for l in LABELS}

                rec = {
                    "scenario_id":   sid,
                    "hazard_type":   scenario["hazard_type"],
                    "ground_truth":  gt,
                    "predicted":     parsed["label"],
                    "confidence":    parsed["confidence"],
                    "class_scores":  cls_scores,
                    "reasoning":     parsed["reasoning"],
                    "parse_ok":      parsed["parse_ok"],
                    "raw_output":    raw_out,
                    "cot_reasoning": cot_stored,
                    "latency_s":     round(latency, 3),
                    "correct":       gt == parsed["label"],
                    "model":         cl_cfg["short_name"],
                    "strategy":      strategy,
                    "cot":           use_cot,
                    "score_source":  "confidence_fallback",
                    "metar_ceiling": scenario["metar"].get("ceiling_ft"),
                    "metar_vis":     scenario["metar"]["visibility_sm"],
                }
                raw_records.append(rec)
                y_true.append(gt); y_pred.append(parsed["label"])
                latencies.append(latency)
                status = "✓" if rec["correct"] else "✗"
                log.info(f"    [{idx+1:3d}/{len(test_set)}] {sid} "
                         f"GT={gt:7s} PRED={parsed['label']:7s} {status} "
                         f"({latency:.2f}s)")

            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_path, "w") as f:
                json.dump({"run_key": run_key, "model": cl_cfg["display"],
                           "strategy": strategy, "cot": use_cot,
                           "timestamp": datetime.now().isoformat(),
                           "n_test": len(test_set),
                           "records": raw_records}, f, indent=2)

            metrics = compute_metrics(y_true, y_pred)
            metrics["avg_latency_s"] = round(np.mean(latencies), 3)
            metrics.update({"model": cl_cfg["display"], "model_key": cl_key,
                            "strategy": strategy, "run_key": run_key,
                            "cot": use_cot})
            all_metrics[run_key]     = metrics
            all_hazard_mets[run_key] = compute_per_hazard_metrics(raw_records)
            log.info(f"    → Acc={metrics['accuracy']:.3f}  "
                     f"F1={metrics['macro_f1']:.3f}  "
                     f"ParseOK={metrics['parse_ok']:.3f}")
            (out_dir / "figures").mkdir(parents=True, exist_ok=True)
            plot_confusion_matrix(metrics["confusion_matrix"],
                                  cl_cfg["short_name"], strategy,
                                  out_dir / "figures" / f"cm_{run_key}.pdf")

    return all_metrics, all_hazard_mets


def save_outputs(all_metrics, all_hazard_mets, out_dir, log):
    met_path = out_dir / "metrics" / "all_metrics.json"
    met_path.parent.mkdir(exist_ok=True)
    with open(met_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"Saved metrics → {met_path}")

    haz_path = out_dir / "metrics" / "per_hazard_metrics.json"
    with open(haz_path, "w") as f:
        json.dump(all_hazard_mets, f, indent=2)

    rows = []
    for key, m in all_metrics.items():
        row = {
            "run_key":   key,
            "model":     m["model"],
            "strategy":  m["strategy"],
            "accuracy":  m["accuracy"],
            "macro_p":   m["macro_p"],
            "macro_r":   m["macro_r"],
            "macro_f1":  m["macro_f1"],
            "latency_s": m.get("avg_latency_s", 0),
            "parse_ok":  m["parse_ok"],
        }
        for lbl in LABELS:
            pc = m["per_class"][lbl]
            row[f"{lbl}_p"]  = pc["precision"]
            row[f"{lbl}_r"]  = pc["recall"]
            row[f"{lbl}_f1"] = pc["f1"]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "metrics" / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved CSV    → {csv_path}")

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_f1_by_model_strategy(all_metrics,  fig_dir / "f1_by_model_strategy.pdf")
    plot_per_class_f1(all_metrics,          fig_dir / "per_class_f1_heatmap.pdf")
    plot_strategy_improvement(all_metrics,  fig_dir / "strategy_improvement.pdf")
    plot_latency(all_metrics,               fig_dir / "latency_comparison.pdf")
    if all_hazard_mets:
        plot_per_hazard_f1(all_hazard_mets, fig_dir / "per_hazard_accuracy.pdf")
    log.info(f"Saved figures → {fig_dir}/")

    tab_dir = out_dir / "tables"
    tab_dir.mkdir(exist_ok=True)
    (tab_dir / "table1_main_results.tex").write_text(
        make_main_results_table(all_metrics))
    (tab_dir / "table2_summary.tex").write_text(
        make_summary_table(all_metrics))
    log.info(f"Saved LaTeX  → {tab_dir}/")

    log.info("\n" + "=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)
    log.info(f"{'Run':<35} {'Acc':>6} {'MacF1':>7} {'Lat(s)':>8}")
    log.info("-" * 60)
    for _, row in df.sort_values("macro_f1", ascending=False).iterrows():
        log.info(
            f"{row['run_key']:<35} "
            f"{row['accuracy']:>6.3f} "
            f"{row['macro_f1']:>7.3f} "
            f"{row['latency_s']:>8.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="CTAF-KHAF LLM evaluation pipeline (open-source + closed-source)"
    )
    parser.add_argument("--dataset",    required=True,
                        help="Path to ctaf_khaf_synthetic_v2.json")
    parser.add_argument("--models",     nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Open-source models to run (default: all three)")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES,
                        choices=STRATEGIES,
                        help="Prompting strategies (default: all three)")
    parser.add_argument("--quantize",   default=None, choices=["4bit", "8bit"],
                        help="Quantization for open-source models (default: fp16)")
    parser.add_argument("--out-dir",    default="results",
                        help="Output root directory (default: ./results)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip inference, regenerate plots from existing metrics")
    parser.add_argument("--closed-source-model",
                        choices=OPENAI_MODEL_KEYS, default=None,
                        metavar="MODEL",
                        help=(
                            "Run a closed-source OpenAI model instead of / in addition to "
                            "open-source models. Choices: " +
                            ", ".join(OPENAI_MODEL_KEYS)
                        ))
    parser.add_argument("--openai-api-key", default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--closed-source-only", action="store_true",
                        help="Skip open-source models, run only the closed-source model")
    parser.add_argument("--cot", action="store_true",
                        help="Use chain-of-thought prompting (two-turn inference). "
                             "Adds '_cot' suffix to run keys so results are stored "
                             "separately from standard runs.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results (ignore resume cache)")
    parser.add_argument("--claude-model",
                        choices=CLAUDE_MODEL_KEYS or None, default=None,
                        metavar="MODEL",
                        help="Run a Claude (Anthropic) model. "
                             "Choices: " + ", ".join(CLAUDE_MODEL_KEYS))
    parser.add_argument("--anthropic-api-key", default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--binary", action="store_true",
                        help="Collapse warning+hazard into 'danger'. Switches the "
                             "system prompt, ICL labels, and label parser to a "
                             "two-class task. Default --out-dir becomes "
                             "results_binary if not explicitly set.")
    args = parser.parse_args()

    # If --binary is requested and the user didn't override --out-dir, route
    # outputs to results_binary/ so the 3-class results stay intact.
    if args.binary and args.out_dir == "results":
        args.out_dir = "results_binary"

    if args.binary:
        apply_binary_mode()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir)

    log.info("CTAF-KHAF Evaluation Pipeline")
    log.info(f"  Mode:       {'BINARY (nominal/danger)' if args.binary else '3-class (nominal/warning/hazard)'}")
    log.info(f"  Models:     {args.models}")
    log.info(f"  Strategies: {args.strategies}")
    log.info(f"  CoT:        {args.cot}")
    log.info(f"  Force:      {args.force}")
    log.info(f"  Output:     {out_dir}/")

    if args.plots_only:
        met_path = out_dir / "metrics" / "all_metrics.json"
        haz_path = out_dir / "metrics" / "per_hazard_metrics.json"
        if not met_path.exists():
            log.error(f"No metrics found at {met_path}. Run without --plots-only first.")
            sys.exit(1)
        with open(met_path)  as f: all_metrics     = json.load(f)
        with open(haz_path)  as f: all_hazard_mets = json.load(f)
        log.info("Regenerating plots from existing metrics...")
        save_outputs(all_metrics, all_hazard_mets, out_dir, log)
        return

    all_metrics, all_hazard_mets = run_experiments(args, log, out_dir)
    save_outputs(all_metrics, all_hazard_mets, out_dir, log)
    log.info("Done.")


if __name__ == "__main__":
    main()