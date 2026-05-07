#!/usr/bin/env python3
"""CTAF-KHAF Synthetic Dataset Generator"""

import os
import sys
import json
import time
import copy
import re
import csv
import random
import argparse
import subprocess
from pathlib import Path
from datetime import timedelta
from collections import Counter


def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q",
                           "--break-system-packages"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def ensure_deps():
    pkgs = ["openai", "faster-whisper", "srt", "pydub", "tqdm"]
    for pkg in pkgs:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"  Installing {pkg}...")
            install(pkg)


AIRPORT   = "KHAF"
RUNWAY    = "30"
PATTERN   = "right"
BASE_LAT  = 37.6194
BASE_LON  = -122.370

CALLSIGNS = [
    "N75WW", "N8AC", "N034CR", "N602SK", "N556KA", "N247GS", "N351KT",
    "N412PB", "N881TF", "N203JL", "N567RA", "N990DX", "N134MV", "N778EH",
    "N405CC", "N223BT", "N689PQ", "N910YZ", "N321FG", "N753HN",
]
AC_TYPES = [
    "Cessna 172", "Piper Cherokee", "Beechcraft Bonanza", "Cirrus SR22",
    "Piper Archer", "Cessna 182", "Mooney M20", "Diamond DA40",
    "Piper Seneca", "Cessna 210",
]

# TTS voice pool — OpenAI has 6 voices; alternate between them for realism
TTS_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# 100 scenarios: 33 nominal, 34 warning, 33 hazard
# Each entry: (hazard_type, safety_label, count)

SCENARIO_PLAN = [
    ("simultaneous_final",        "hazard",  9),
    ("wrong_runway_announcement", "hazard",  8),
    ("imc_vfr_conflict",          "hazard",  8),
    ("runway_incursion_risk",      "hazard",  8),
    ("missing_position_calls",    "warning", 7),
    ("pattern_conflict",          "warning", 7),
    ("silent_traffic",            "warning", 7),
    ("go_around_conflict",        "warning", 7),
    ("improper_entry",            "warning", 6),
    ("nominal_single_aircraft",       "nominal", 11),
    ("nominal_multi_aircraft",        "nominal", 11),
    ("nominal_instrument_approach",   "nominal", 11),
]

LABEL_METAR = {
    "nominal": ["clear_vfr", "windy_vfr", "marginal_vfr"],
    "warning": ["clear_vfr", "marginal_vfr", "overcast_imc", "windy_vfr"],
    "hazard":  ["low_imc", "overcast_imc", "marginal_vfr", "clear_vfr"],
}

METAR_DEFS = {
    "clear_vfr": {
        "template": "KHAF {Z} AUTO {W} 10SM CLR {T}/{D} A{A} RMK AO2",
        "description": "Clear skies, 10 SM visibility, excellent VFR conditions",
        "ceiling_ft": None, "visibility_sm": 10, "wind_kt_range": (5, 15),
    },
    "marginal_vfr": {
        "template": "KHAF {Z} AUTO {W} 5SM -BR FEW010 BKN020 {T}/{D} A{A} RMK AO2",
        "description": "Marginal VFR, 5 SM visibility in mist, broken ceiling at 2,000 ft",
        "ceiling_ft": 2000, "visibility_sm": 5, "wind_kt_range": (5, 10),
    },
    "low_imc": {
        "template": "KHAF {Z} AUTO {W} 1SM -FG OVC004 {T}/{D} A{A} RMK AO2",
        "description": "IMC conditions, 1 SM visibility in fog, overcast at 400 ft",
        "ceiling_ft": 400, "visibility_sm": 1, "wind_kt_range": (3, 8),
    },
    "overcast_imc": {
        "template": "KHAF {Z} AUTO {W} 9SM OVC016 {T}/{D} A{A} RMK AO2",
        "description": "Overcast ceiling at 1,600 ft, 9 SM visibility, IFR conditions",
        "ceiling_ft": 1600, "visibility_sm": 9, "wind_kt_range": (5, 15),
    },
    "windy_vfr": {
        "template": "KHAF {Z} AUTO {W} 8SM SCT015 BKN030 {T}/{D} A{A} RMK AO2",
        "description": "Moderate winds, 8 SM visibility, scattered clouds, VFR",
        "ceiling_ft": 3000, "visibility_sm": 8, "wind_kt_range": (15, 25),
    },
}

# Missing-call ground truth (used in dataset metadata)
MISSING_CALLS_MAP = {
    "missing_position_calls": ["downwind"],
    "silent_traffic":         ["all_calls_cs2"],
    "go_around_conflict":     ["go_around_announcement"],
    "runway_incursion_risk":  ["runway_clear_announcement"],
}


def rnd(a, b):   return random.randint(a, b)
def pick(lst):   return random.choice(lst)

def build_metar(key: str) -> dict:
    d   = METAR_DEFS[key]
    wkt = rnd(*d["wind_kt_range"])
    wdr = pick([180, 210, 240, 270, 300, 330, 360])
    t   = rnd(12, 18)
    dp  = t - rnd(1, 4)
    alt = pick([2995, 2999, 3001, 3005])
    day = rnd(1, 28)
    hr  = rnd(14, 22)
    mn  = pick([55, 35, 15])
    z   = f"{day:02d}{hr:02d}{mn:02d}Z"
    w   = f"{wdr:03d}{wkt:02d}KT"
    raw = (d["template"]
           .replace("{Z}", z).replace("{W}", w)
           .replace("{T}", str(t)).replace("{D}", str(dp))
           .replace("{A}", str(alt)))
    return {
        "raw": raw,
        "description": d["description"],
        "ceiling_ft": d["ceiling_ft"],
        "visibility_sm": d["visibility_sm"],
        "wind_direction": wdr,
        "wind_kt": wkt,
        "temperature_c": t,
        "dewpoint_c": dp,
    }


TRANSCRIPT_PROMPTS = {

"simultaneous_final": lambda cs1,t1,cs2,t2: f"""
Generate a realistic CTAF radio transcript for Half Moon Bay Airport (KHAF), runway 30, right traffic.
SCENARIO: Two aircraft are simultaneously converging on final approach for runway 30 with no coordination.
  - {cs1} ({t1}): On a 2-mile straight-in RNAV final for runway 30; does not see the pattern traffic.
  - {cs2} ({t2}): Turned final from the pattern without detecting {cs1} on the instrument approach.
  - Neither acknowledges the other until dangerously close. Show escalating urgency and a near mid-air collision.
  - One aircraft must call go-around at the last moment.
REQUIREMENTS:
  - 10–14 SRT entries with realistic timestamps.
  - Authentic CTAF format: "[Airport] traffic, [callsign], [position], [intentions], [airport]."
  - Vary urgency: early calls are routine, later calls show alarm.
  - Output ONLY the SRT block. No preamble, no explanation.
""",

"missing_position_calls": lambda cs1,t1,cs2,t2: f"""
Generate a realistic CTAF radio transcript for Half Moon Bay Airport (KHAF), runway 30, right traffic.
SCENARIO: {cs1} ({t1}) enters the traffic pattern but omits the downwind position call entirely — going
  directly from crosswind to base with no downwind announcement. {cs2} ({t2}) is on final and unaware
  of {cs1}'s exact position in the pattern.
REQUIREMENTS:
  - 8–12 SRT entries.
  - The gap must be clear: {cs1} makes a crosswind call, then immediately a base call — no downwind.
  - {cs2} explicitly questions traffic position and sounds concerned.
  - Output ONLY the SRT block.
""",

"pattern_conflict": lambda cs1,t1,cs2,t2: f"""
Generate a realistic CTAF radio transcript for Half Moon Bay Airport (KHAF), runway 30, right traffic.
SCENARIO: {cs1} ({t1}) is on base leg. {cs2} ({t2}) is on a long final for runway 30. They are
  converging and both realise it. They communicate, and one extends to give the other spacing.
REQUIREMENTS:
  - 10–14 SRT entries.
  - Both aircraft make all standard pattern calls throughout.
  - Show the conflict identification, negotiation, and resolution.
  - One aircraft says "extending final" or "I'll follow you."
  - Output ONLY the SRT block.
""",

"wrong_runway_announcement": lambda cs1,t1,cs2,t2: f"""
Generate a realistic CTAF radio transcript for Half Moon Bay Airport (KHAF).
SCENARIO: {cs1} ({t1}) erroneously announces for runway 12 (the reciprocal of the active runway 30).
  {cs2} ({t2}) is on a half-mile final for runway 30 and hears the incorrect call.
  {cs2} urgently alerts {cs1}, who must break off and re-sequence.
REQUIREMENTS:
  - 8–12 SRT entries.
  - Show the wrong-runway call clearly, the correction broadcast from {cs2}, and {cs1}'s recovery.
  - Output ONLY the SRT block.
""",

"imc_vfr_conflict": lambda cs1,t1,cs2,t2: f"""
Generate a realistic CTAF radio transcript for Half Moon Bay Airport (KHAF), runway 30.
SCENARIO: Weather is IMC — overcast at 400 ft, 1 SM fog. {cs1} ({t1}) is a VFR-only pilot who has
  inadvertently entered IMC on approach. Their calls reflect spatial disorientation: wrong altitude,
  inconsistent position reports, hesitant transmissions. {cs2} ({t2}) is IFR-certified and on the
  RNAV 30 approach; they recognise the danger and warn {cs1} urgently.
REQUIREMENTS:
  - 8–12 SRT entries.
  - {cs1}'s calls should sound uncertain/confused (altitude errors, unclear position).
  - {cs2} calls out the danger explicitly and advises {cs1} to climb immediately.
  - Output ONLY the SRT block.
""",

"silent_traffic": lambda cs1,t1,cs2,t2: f"""
Generate a realistic CTAF radio transcript for Half Moon Bay Airport (KHAF), runway 30, right traffic.
SCENARIO: {cs1} ({t1}) is flying the standard pattern and broadcasting all calls correctly. ADS-B shows
  a second aircraft ({cs2}) at 1,200 ft about 1.5 NM from the field, but {cs2} makes ZERO radio calls
  (NORDO — no radio). {cs1} calls out the blind traffic multiple times with growing alarm.
REQUIREMENTS:
  - 8–12 SRT entries.
  - {cs1} makes all standard calls AND repeatedly calls out the unknown traffic.
  - Urgency increases with each "blind traffic" call.
  - {cs2} never responds (silence is part of the scenario).
  - Output ONLY the SRT block.
""",

"go_around_conflict": lambda cs1,t1,cs2,t2: f"""
Generate a realistic CTAF radio transcript for Half Moon Bay Airport (KHAF), runway 30.
SCENARIO: {cs1} ({t1}) is on short final and executes a go-around but does NOT announce it on CTAF.
  {cs2} ({t2}), lined up for departure on runway 30, begins its takeoff roll. {cs1} is climbing out
  directly over the departure end of the runway while {cs2} is accelerating. {cs2} calls out the
  conflict. Only then does {cs1} belatedly announce the go-around.
REQUIREMENTS:
  - 10–14 SRT entries.
  - {cs1}'s go-around is NOT announced until after {cs2} calls out the conflict.
  - {cs2}'s alarm must be explicit ("traffic on runway!").
  - Output ONLY the SRT block.
""",

"improper_entry": lambda cs1,t1,cs2,t2: f"""
Generate a realistic CTAF radio transcript for Half Moon Bay Airport (KHAF), runway 30, right traffic.
SCENARIO: {cs1} ({t1}) approaches from the west and announces a straight-in final approach without
  entering the standard 45-degree downwind entry. {cs2} ({t2}) is on the right downwind and questions
  the non-standard entry. They must coordinate to avoid a conflict.
REQUIREMENTS:
  - 8–12 SRT entries.
  - {cs1}'s calls describe a straight-in approach (no crosswind/downwind leg).
  - {cs2} calls out the traffic and requests {cs1} state intentions.
  - Show resolution: one aircraft adjusts.
  - Output ONLY the SRT block.
""",

"runway_incursion_risk": lambda cs1,t1,cs2,t2: f"""
Generate a realistic CTAF radio transcript for Half Moon Bay Airport (KHAF), runway 30.
SCENARIO: {cs1} ({t1}) has just landed and is rolling out on runway 30 but does NOT announce clearing
  the runway. {cs2} ({t2}) is 0.5 NM short final and makes repeated calls asking for runway status.
  {cs1} does not respond (radio distraction). {cs2} eventually executes a precautionary go-around.
REQUIREMENTS:
  - 8–12 SRT entries.
  - {cs2} makes at least 3 calls asking if the runway is clear.
  - {cs1} is completely silent throughout.
  - {cs2}'s final call announces the go-around.
  - Output ONLY the SRT block.
""",

"nominal_single_aircraft": lambda cs1,t1,cs2,t2: f"""
Generate a realistic CTAF radio transcript for Half Moon Bay Airport (KHAF), runway 30, right traffic.
SCENARIO: A single aircraft {cs1} ({t1}) flies a complete, textbook traffic pattern — one call per leg:
  10-mile inbound, entering downwind, crosswind, downwind, base, final, and clearing the runway.
  Excellent, professional communications throughout. No other traffic.
REQUIREMENTS:
  - 7–9 SRT entries (one per leg plus inbound and runway-clear).
  - Use correct CTAF phrasing. No errors. No other traffic.
  - Output ONLY the SRT block.
""",

"nominal_multi_aircraft": lambda cs1,t1,cs2,t2: f"""
Generate a realistic CTAF radio transcript for Half Moon Bay Airport (KHAF), runway 30, right traffic.
SCENARIO: Two aircraft — {cs1} ({t1}) and {cs2} ({t2}) — are in the right traffic pattern. {cs2} is
  ahead of {cs1} by exactly one leg. Both make all required position calls. They acknowledge each other
  naturally ("traffic in sight") and maintain safe separation throughout. Both land without incident.
REQUIREMENTS:
  - 12–16 interleaved SRT entries.
  - Both aircraft make all standard pattern calls (crosswind, downwind, base, final, clear).
  - Include at least two acknowledgements of each other's traffic.
  - Output ONLY the SRT block.
""",

"nominal_instrument_approach": lambda cs1,t1,cs2,t2: f"""
Generate a realistic CTAF radio transcript for Half Moon Bay Airport (KHAF), runway 30.
SCENARIO: {cs1} ({t1}) executes a clean RNAV 30 instrument approach, announcing distance and altitude
  at standard reporting points (10 NM, 7 NM, 5 NM, 3 NM, and short final). {cs2} ({t2}) is in the VFR
  right-hand traffic pattern. Both communicate properly, identify each other, coordinate safely, and
  both land without incident.
REQUIREMENTS:
  - 12–16 interleaved SRT entries.
  - {cs1} uses instrument approach phraseology (distance from field, altitude, "straight in RNAV 30").
  - {cs2} makes all standard VFR pattern calls and acknowledges the IFR traffic.
  - Output ONLY the SRT block.
""",
}


def advisory_prompt(hazard_type, label, cs1, t1, cs2, t2, metar_desc, tx_lines):
    sample = " | ".join(tx_lines[:6])
    return f"""You are an expert aviation safety analyst at a non-towered airport.

Scenario at Half Moon Bay Airport (KHAF):
  - Hazard type : {hazard_type}
  - Safety label: {label}
  - Aircraft    : {cs1} ({t1})  and  {cs2} ({t2})
  - Weather     : {metar_desc}
  - Transcript  : {sample}

Write a concise 2–3 sentence ground-truth advisory in FAA-conforming phraseology that an automated
safety system should output for this scenario. Be specific: reference callsigns, positions, and
recommended actions. If the scenario is nominal, state that operations appear normal and no corrective
advisory is required.

Output ONLY the advisory text. No preamble.
"""


def parse_srt(raw: str) -> list[dict]:
    entries = []
    lines   = raw.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r"^\d+$", line):
            idx = int(line)
            i += 1
            if i >= len(lines):
                break
            timestamp = lines[i].strip()
            i += 1
            parts = []
            while i < len(lines) and lines[i].strip() != "" and not re.match(r"^\d+$", lines[i].strip()):
                parts.append(lines[i].strip())
                i += 1
            if parts:
                entries.append({"index": idx, "timestamp": timestamp,
                                 "text": " ".join(parts)})
            while i < len(lines) and lines[i].strip() == "":
                i += 1
        else:
            i += 1
    return entries


def build_tts_script(entries: list[dict], cs1: str, cs2: str) -> list[dict]:
    """Assign a TTS voice to each transcript line based on the callsign that"""
    voice_cs1 = pick(TTS_VOICES[:3])   # alloy / echo / fable
    voice_cs2 = pick(TTS_VOICES[3:])   # onyx / nova / shimmer

    script = []
    for e in entries:
        txt = e["text"]
        # Determine speaker
        if cs1.replace("N","").lower() in txt.lower() or cs1.lower() in txt.lower():
            voice = voice_cs1
        elif cs2.replace("N","").lower() in txt.lower() or cs2.lower() in txt.lower():
            voice = voice_cs2
        else:
            # Interleave for realism
            voice = voice_cs1 if len(script) % 2 == 0 else voice_cs2
        # Add a tiny silence prefix to simulate radio squelch open
        script.append({"voice": voice, "text": txt, "timestamp": e["timestamp"]})
    return script


PHONETIC = {
    "alpha":"A","bravo":"B","charlie":"C","delta":"D","echo":"E",
    "foxtrot":"F","golf":"G","hotel":"H","india":"I","juliet":"J",
    "kilo":"K","lima":"L","mike":"M","november":"N","oscar":"O",
    "papa":"P","quebec":"Q","romeo":"R","sierra":"S","tango":"T",
    "uniform":"U","victor":"V","whiskey":"W","xray":"X",
    "yankee":"Y","zulu":"Z",
}
_PHON_PAT = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in PHONETIC) + r")\b",
    re.IGNORECASE,
)
def translate_phonetic(text: str) -> str:
    return _PHON_PAT.sub(lambda m: PHONETIC[m.group(0).lower()], text)


def build_adsb(cs1, t1, cs2, t2, hazard_type):
    states = []
    for i, (cs, ac) in enumerate([(cs1, t1), (cs2, t2)]):
        states.append({
            "callsign":     cs,
            "aircraft_type": ac,
            "latitude":     round(BASE_LAT + random.uniform(-0.15, 0.15), 4),
            "longitude":    round(BASE_LON + random.uniform(-0.15, 0.15), 4),
            "altitude_ft":  rnd(800, 3500),
            "heading":      pick([30, 120, 210, 300, 270, 90]),
            "velocity_kt":  rnd(65, 120),
            "has_radio":    not (hazard_type == "silent_traffic" and i == 1),
            "on_ground":    False,
        })
    return states


def generate_scenario(
    client,           # openai.OpenAI instance
    scenario_id: str,
    hazard_type: str,
    label: str,
    metar_key: str,
    out_dir: Path,
    whisper_model,    # faster_whisper.WhisperModel instance (or None)
    gpt_model: str,
    tts_model: str,
    verbose: bool,
) -> dict:

    cs_pool = random.sample(CALLSIGNS, 2)
    ac_pool = random.sample(AC_TYPES, 2)
    cs1, cs2 = cs_pool
    t1, t2   = ac_pool

    metar = build_metar(metar_key)

    tx_prompt = TRANSCRIPT_PROMPTS[hazard_type](cs1, t1, cs2, t2)
    tx_resp = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system",
             "content": ("You are an expert general aviation radio communications simulator. "
                         "You produce highly realistic, authentic CTAF radio transcripts in SRT format. "
                         "Output ONLY valid SRT content — no commentary, no markdown fences.")},
            {"role": "user", "content": tx_prompt},
        ],
        temperature=0.9,
        max_completion_tokens=1400,
    )
    tx_raw = tx_resp.choices[0].message.content.strip()
    # Strip any markdown fences GPT might add
    tx_raw = re.sub(r"```[a-z]*\n?", "", tx_raw).strip()

    gt_entries = parse_srt(tx_raw)
    if not gt_entries:
        raise ValueError(f"SRT parse returned 0 entries for {scenario_id}")

    # Save ground-truth SRT
    sc_dir = out_dir / scenario_id
    sc_dir.mkdir(parents=True, exist_ok=True)
    gt_srt_path = sc_dir / "transcript_ground_truth.srt"
    gt_srt_path.write_text(tx_raw, encoding="utf-8")

    tx_lines = [e["text"] for e in gt_entries]
    adv_prompt = advisory_prompt(hazard_type, label, cs1, t1, cs2, t2,
                                  metar["description"], tx_lines)
    adv_resp = client.chat.completions.create(
        model=gpt_model,
        messages=[{"role": "user", "content": adv_prompt}],
        temperature=0.5,
        max_completion_tokens=300,
    )
    advisory = adv_resp.choices[0].message.content.strip()

    # Build a per-voice script and concatenate audio chunks
    tts_script  = build_tts_script(gt_entries, cs1, cs2)
    audio_chunks = []
    audio_path   = sc_dir / "audio.mp3"

    for item in tts_script:
        try:
            tts_resp = client.audio.speech.create(
                model=tts_model,
                voice=item["voice"],
                input=item["text"],
                response_format="mp3",
            )
            audio_chunks.append(tts_resp.content)
        except Exception as e:
            if verbose:
                print(f"      [TTS] chunk failed: {e}")

    if audio_chunks:
        # Simple concatenation of MP3 frames (works for sequential speech)
        # For higher quality, pydub can be used to add silence between chunks
        try:
            from pydub import AudioSegment
            import io
            silence_100ms = AudioSegment.silent(duration=350)   # 350 ms squelch gap
            combined = AudioSegment.empty()
            for chunk in audio_chunks:
                seg = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                combined += seg + silence_100ms
            combined.export(str(audio_path), format="mp3")
            audio_generated = True
        except Exception as e:
            # Fallback: raw concatenation
            with open(audio_path, "wb") as f:
                for chunk in audio_chunks:
                    f.write(chunk)
            audio_generated = True
    else:
        audio_generated = False

    whisper_entries = []
    whisper_raw     = ""
    if whisper_model is not None and audio_generated:
        try:
            segments, info = whisper_model.transcribe(
                str(audio_path),
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.6,
                    "min_speech_duration_ms": 250,
                    "max_speech_duration_s": float("inf"),
                    "min_silence_duration_ms": 100,
                    "speech_pad_ms": 400,
                },
            )
            # Merge words into sentence segments (mirrors whisper.py logic)
            word_segs = []
            for seg in segments:
                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        if w.word.strip():
                            word_segs.append({"text": w.word.strip(),
                                              "start": w.start, "end": w.end})

            # Sentence merger (mirrors whisper.py sentence_segments_merger)
            merged = _merge_segments(word_segs, max_len=80, max_gap=2.0)

            # Build SRT
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
            whisper_raw = srt_lib.compose(srt_list)
            whisper_entries = parse_srt(whisper_raw)

            # Save whisper SRT
            whisper_srt_path = sc_dir / "transcript_whisper.srt"
            whisper_srt_path.write_text(whisper_raw, encoding="utf-8")

            # Save phonetic-translated plain text (mirrors whisper.py output)
            translated = translate_phonetic(whisper_raw)
            whisper_txt_path = sc_dir / "transcript_whisper.txt"
            whisper_txt_path.write_text(translated, encoding="utf-8")

        except Exception as e:
            if verbose:
                print(f"      [Whisper] failed: {e}")

    scenario = {
        "scenario_id":   scenario_id,
        "label":         label,
        "hazard_type":   hazard_type,
        "airport":       AIRPORT,
        "runway":        RUNWAY,
        "traffic_pattern": PATTERN,
        "metar":         metar,
        "adsb_states":   build_adsb(cs1, t1, cs2, t2, hazard_type),
        "aircraft": [
            {"callsign": cs1, "type": t1},
            {"callsign": cs2, "type": t2},
        ],
        # Ground-truth transcript (GPT-generated)
        "transcript_ground_truth":     gt_entries,
        "transcript_ground_truth_raw": tx_raw,
        # Whisper-re-transcribed transcript (from TTS audio)
        "transcript_whisper":          whisper_entries,
        "transcript_whisper_raw":      whisper_raw,
        # Audio file (relative path)
        "audio_file": str(sc_dir / "audio.mp3") if audio_generated else None,
        # Safety metadata
        "missing_calls":          MISSING_CALLS_MAP.get(hazard_type, []),
        "ground_truth_advisory":  advisory,
        "num_gt_entries":         len(gt_entries),
        "num_whisper_entries":    len(whisper_entries),
        "audio_generated":        audio_generated,
        "whisper_transcribed":    len(whisper_entries) > 0,
    }
    return scenario


def _merge_segments(segs, max_len=80, max_gap=2.0):
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
        gap = s["start"] - cur["end"]
        combined = cur["text"] + " " + txt
        if gap < max_gap and len(combined) < max_len:
            cur["text"] = combined.strip()
            cur["end"]  = s["end"]
        else:
            merged.append(copy.deepcopy(cur))
            cur = {"text": txt, "start": s["start"], "end": s["end"]}
    if cur:
        merged.append(cur)
    return merged


def build_plan(seed=42) -> list[dict]:
    random.seed(seed)
    plan = []
    sid  = 1
    for hazard_type, label, count in SCENARIO_PLAN:
        for _ in range(count):
            metar_key = pick(LABEL_METAR[label])
            plan.append({
                "id":          f"S{sid:03d}",
                "hazard_type": hazard_type,
                "label":       label,
                "metar_key":   metar_key,
            })
            sid += 1
    # Shuffle so scenarios aren't in hazard-type blocks
    random.shuffle(plan)
    return plan[:100]


def write_csv(dataset: list[dict], path: Path):
    fields = [
        "scenario_id", "label", "hazard_type",
        "metar_ceiling_ft", "metar_visibility_sm", "metar_wind_kt",
        "cs1", "cs2", "type1", "type2",
        "num_gt_entries", "num_whisper_entries",
        "audio_generated", "whisper_transcribed",
        "missing_calls",
        "ground_truth_advisory",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in dataset:
            w.writerow({
                "scenario_id":       s["scenario_id"],
                "label":             s["label"],
                "hazard_type":       s["hazard_type"],
                "metar_ceiling_ft":  s["metar"].get("ceiling_ft", ""),
                "metar_visibility_sm": s["metar"]["visibility_sm"],
                "metar_wind_kt":     s["metar"]["wind_kt"],
                "cs1":               s["aircraft"][0]["callsign"],
                "cs2":               s["aircraft"][1]["callsign"],
                "type1":             s["aircraft"][0]["type"],
                "type2":             s["aircraft"][1]["type"],
                "num_gt_entries":    s["num_gt_entries"],
                "num_whisper_entries": s["num_whisper_entries"],
                "audio_generated":   s["audio_generated"],
                "whisper_transcribed": s["whisper_transcribed"],
                "missing_calls":     "|".join(s["missing_calls"]),
                "ground_truth_advisory": s["ground_truth_advisory"],
            })


def main():
    parser = argparse.ArgumentParser(
        description="Generate CTAF-KHAF synthetic dataset (GPT + TTS + Whisper)"
    )
    parser.add_argument("--api-key",    required=True,
                        help="OpenAI API key")
    parser.add_argument("--out-dir",    default="dataset",
                        help="Root output directory (default: ./dataset)")
    parser.add_argument("--gpt-model",  default="gpt-4.1",
                        help="GPT model for transcript & advisory generation (default: gpt-4.1)")
    parser.add_argument("--tts-model",  default="tts-1-hd",
                        help="OpenAI TTS model (default: tts-1-hd)")
    parser.add_argument("--whisper-model", default="large-v3",
                        help="faster-whisper model size (default: large-v3). "
                             "Use 'none' to skip Whisper.")
    parser.add_argument("--start",      type=int, default=1,
                        help="Start at scenario N (1-indexed, for resuming)")
    parser.add_argument("--end",        type=int, default=100,
                        help="End at scenario N inclusive (default: 100)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--delay",      type=float, default=0.5,
                        help="Seconds to sleep between API calls (default: 0.5)")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  CTAF-KHAF Synthetic Dataset Generator")
    print("=" * 60)
    print(f"  GPT model    : {args.gpt_model}")
    print(f"  TTS model    : {args.tts_model}")
    print(f"  Whisper model: {args.whisper_model}")
    print(f"  Output dir   : {args.out_dir}")
    print(f"  Range        : S{args.start:03d} – S{args.end:03d}")
    print("=" * 60)

    print("\nChecking dependencies...")
    ensure_deps()

    import openai
    client = openai.OpenAI(api_key=args.api_key)

    whisper_model = None
    if args.whisper_model.lower() != "none":
        print(f"\nLoading faster-whisper ({args.whisper_model})...")
        try:
            from faster_whisper import WhisperModel
            device       = "cuda" if _cuda_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            whisper_model = WhisperModel(args.whisper_model,
                                         device=device,
                                         compute_type=compute_type)
            print(f"  Whisper loaded on {device}.")
        except Exception as e:
            print(f"  [WARNING] Could not load Whisper: {e}")
            print("  Audio will be generated but not re-transcribed.")

    plan    = build_plan(seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sc_dir  = out_dir / "scenarios"
    sc_dir.mkdir(exist_ok=True)

    # Load existing dataset if resuming
    json_path = out_dir / "ctaf_khaf_synthetic_v1.json"
    if json_path.exists() and args.start > 1:
        with open(json_path) as f:
            existing = json.load(f)
        dataset = existing.get("scenarios", [])
        print(f"\nResuming — loaded {len(dataset)} existing scenarios.")
    else:
        dataset = []

    counters = Counter(s["label"] for s in dataset)

    print(f"\nGenerating scenarios {args.start}–{args.end}...\n")

    for i, item in enumerate(plan[args.start - 1: args.end], start=args.start):
        sid  = item["id"]
        bar  = f"[{i:3d}/{args.end}]"
        lbl  = item["label"].upper()[:4]
        print(f"{bar} {sid} | {item['hazard_type']:<30} | {lbl:<4} | {item['metar_key']}", end="  ")
        sys.stdout.flush()

        try:
            sc = generate_scenario(
                client       = client,
                scenario_id  = sid,
                hazard_type  = item["hazard_type"],
                label        = item["label"],
                metar_key    = item["metar_key"],
                out_dir      = sc_dir,
                whisper_model= whisper_model,
                gpt_model    = args.gpt_model,
                tts_model    = args.tts_model,
                verbose      = args.verbose,
            )
            dataset.append(sc)
            counters[item["label"]] += 1
            audio_ok   = "A✓" if sc["audio_generated"]    else "A✗"
            whisper_ok = "W✓" if sc["whisper_transcribed"] else "W✗"
            print(f"✓  GT:{sc['num_gt_entries']:2d} {audio_ok} {whisper_ok}")

        except Exception as e:
            print(f"✗  {type(e).__name__}: {str(e)[:80]}")

        # Checkpoint save every 5 scenarios
        if i % 5 == 0 or i == args.end:
            _save_dataset(
                dataset,
                json_path,
                out_dir,
                counters,
                args.gpt_model,
                args.tts_model,
                args.whisper_model,
            )

        time.sleep(args.delay)

    _save_dataset(
        dataset,
        json_path,
        out_dir,
        counters,
        args.gpt_model,
        args.tts_model,
        args.whisper_model,
    )
    write_csv(dataset, out_dir / "dataset_summary.csv")

    print("\n" + "=" * 60)
    print(f"  Done.  {len(dataset)} scenarios saved to {out_dir}/")
    print(f"  nominal : {counters['nominal']:3d}")
    print(f"  warning : {counters['warning']:3d}")
    print(f"  hazard  : {counters['hazard']:3d}")
    print(f"  Audio generated      : {sum(1 for s in dataset if s['audio_generated'])}")
    print(f"  Whisper transcribed  : {sum(1 for s in dataset if s['whisper_transcribed'])}")
    print("=" * 60)


def _save_dataset(dataset, json_path, out_dir, counters, gpt_model, tts_model, whisper_model):
    from collections import Counter as C
    htypes = C(s["hazard_type"] for s in dataset)
    output = {
        "metadata": {
            "name":        "CTAF-KHAF-Synthetic-v1",
            "description": (
                "Synthetic CTAF radio transcript dataset for evaluating VLM-based "
                "aviation safety analysis at non-towered airports. "
                "Generated for Half Moon Bay Airport (KHAF), runway 30."
            ),
            "airport":     AIRPORT,
            "airport_lat": BASE_LAT,
            "airport_lon": BASE_LON,
            "runway":      RUNWAY,
            "traffic_pattern": PATTERN,
            "total_scenarios": len(dataset),
            "label_distribution": {
                "nominal": counters.get("nominal", 0),
                "warning": counters.get("warning", 0),
                "hazard":  counters.get("hazard",  0),
            },
            "hazard_type_distribution": dict(htypes),
            "generation_model_transcript": gpt_model,
            "generation_model_tts":        tts_model,
            "asr_model":                   whisper_model,
            "reference_paper": (
                "Darrell et al., AIAA 2026 — Automated Multimodal Analysis of "
                "Air Traffic around Non-Towered Airports using Large Language Models"
            ),
        },
        "label_schema": {
            "nominal": "Normal operations — all required communications present, no conflicts",
            "warning": "Potential conflict or communication gap — recoverable with pilot action",
            "hazard":  "Imminent safety risk — immediate action required to prevent collision",
        },
        "hazard_type_schema": {
            ht: lbl for ht, lbl, _ in SCENARIO_PLAN
        },
        "pipeline": {
            "step_1": f"{gpt_model} generates SRT-formatted CTAF transcript + advisory",
            "step_2": f"OpenAI TTS ({tts_model}) synthesises per-line MP3 audio with distinct voices per aircraft",
            "step_3": f"faster-whisper ({whisper_model}) re-transcribes audio → SRT (closes the ASR loop)",
            "step_4": "NATO phonetic alphabet translated in whisper.txt output",
        },
        "scenarios": dataset,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def _cuda_available():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    main()
