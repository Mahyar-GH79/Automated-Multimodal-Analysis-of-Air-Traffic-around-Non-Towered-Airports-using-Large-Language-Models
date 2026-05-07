#!/usr/bin/env python3
"""CTAF-KHAF Dataset Postprocessor"""

import json
import copy
import random
import argparse
import re
from pathlib import Path
from datetime import datetime


# These are replacement METARs for the two imc_vfr_conflict mismatches.
# We preserve the original timestamp and wind so only condition changes.

IMC_REPLACEMENT = {
    # S019: was KHAF 171735Z AUTO 18006KT 10SM CLR 18/15 A2995 RMK AO2
    # Transcript has: fog, imc, cloud, overcast, soup → needs low_imc
    "S019": {
        "raw":            "KHAF 171735Z AUTO 18006KT 1SM -FG OVC004 18/17 A2995 RMK AO2",
        "description":    "IMC conditions, 1 SM visibility in fog, overcast at 400 ft — KHAF coastal fog",
        "ceiling_ft":     400,
        "visibility_sm":  1,
        "wind_direction": 180,
        "wind_kt":        6,
        "temperature_c":  18,
        "dewpoint_c":     17,
    },
    # S023: was KHAF 151455Z AUTO 36009KT 10SM CLR 15/13 A3005 RMK AO2
    # Transcript has: fog, imc → needs low_imc
    "S023": {
        "raw":            "KHAF 151455Z AUTO 36009KT 1SM -FG OVC004 15/14 A3005 RMK AO2",
        "description":    "IMC conditions, 1 SM visibility in fog, overcast at 400 ft — KHAF coastal fog",
        "ceiling_ft":     400,
        "visibility_sm":  1,
        "wind_direction": 360,
        "wind_kt":        9,
        "temperature_c":  15,
        "dewpoint_c":     14,
    },
}


GEO_PHRASES = {

    # For inbound / first call — pilot reporting position relative to local geography
    "inbound": [
        "crossing the coastline inbound",
        "offshore, crossing Pillar Point",
        "over the ocean, three miles west",
        "crossing the coast south of Princeton",
        "over the water, entering the Half Moon Bay area",
        "just inside the coast, southeast of the field",
        "tracking in from offshore over Pillar Point Harbor",
        "west of Montara, inbound",
        "south of Moss Beach, inbound",
        "over the coastline, El Granada area",
    ],

    # For pattern calls — situational awareness of the coastal environment
    "pattern": [
        "have the ocean in sight to the west",
        "Pillar Point visible off my right wing",
        "coastal fog bank staying offshore",
        "marine layer holding south of the field",
        "clear of the coastal hills",
        "Montara Mountain in sight",
        "over the Princeton area",
        "keeping clear of the Pillar Point restricted area",
    ],

    # For AWOS / weather remarks
    "weather": [
        "AWOS on 127.275 shows",
        "KHAF AWOS indicating",
        "NorCal Approach on 135.1 advising",
        "marine layer moving onshore",
        "coastal fog advancing from the west",
        "typical Half Moon Bay marine layer",
        "fog bank just offshore of Pillar Point",
    ],

    # For go-around / conflict calls — local terrain awareness
    "terrain": [
        "staying clear of Montara Mountain to the north",
        "climbing away from terrain to the east",
        "remaining over water until established",
        "clear of coastal hills",
    ],

    # For advisory text — controller-style local context
    "advisory_local": [
        "at Half Moon Bay Airport (KHAF), coastal environment",
        "KHAF, runway 30, right traffic, coastal airport",
        "at KHAF — be aware of marine layer and coastal terrain",
        "Half Moon Bay Airport — coastal fog advisory in effect",
        "KHAF coastal airport — Montara Mountain terrain 1,800 ft MSL to the northeast",
        "at KHAF — NorCal Approach available on 135.1 if IFR assistance needed",
    ],
}

# Which hazard types get which phrase types injected
HAZARD_PHRASE_STRATEGY = {
    # Inbound mention → add coastal inbound context to first call
    # Pattern call    → add local landmark to mid-pattern call
    "simultaneous_final":        ["inbound", "pattern"],
    "missing_position_calls":    ["inbound", "pattern"],
    "pattern_conflict":          ["inbound", "pattern"],
    "wrong_runway_announcement": ["inbound", "pattern"],
    "imc_vfr_conflict":          ["weather", "terrain"],
    "silent_traffic":            ["inbound", "pattern"],
    "go_around_conflict":        ["terrain", "pattern"],
    "improper_entry":            ["inbound", "pattern"],
    "runway_incursion_risk":     ["inbound", "pattern"],
    "nominal_single_aircraft":   ["inbound", "pattern"],
    "nominal_multi_aircraft":    ["inbound", "pattern"],
    "nominal_instrument_approach": ["inbound", "weather"],
}


def _already_has_geo(text: str) -> bool:
    """Return True if the entry already contains a KHAF-specific geographic reference."""
    markers = [
        "pillar point", "princeton", "montara", "el granada", "moss beach",
        "norcal", "127.275", "135.1", "awos", "coastline", "offshore",
        "marine layer", "coastal fog", "ocean", "over the water",
    ]
    tl = text.lower()
    return any(m in tl for m in markers)


def _find_injection_entry(entries: list, strategy: list) -> tuple[int, str]:
    """Return (entry_index, phrase_type) for the best injection point."""
    # Try each phrase type in strategy order
    for phrase_type in strategy:
        if phrase_type in ("inbound", "weather"):
            # Inject into first entry that mentions miles/inbound/southeast
            for i, e in enumerate(entries):
                tl = e["text"].lower()
                if any(x in tl for x in ["miles", "inbound", "southeast", "northwest", "southwest", "west of", "east of"]):
                    if not _already_has_geo(e["text"]):
                        return i, phrase_type
        elif phrase_type in ("pattern", "terrain"):
            # Inject into first downwind/base entry
            for i, e in enumerate(entries):
                tl = e["text"].lower()
                if any(x in tl for x in ["downwind", "base", "crosswind"]):
                    if not _already_has_geo(e["text"]):
                        return i, phrase_type

    # Fallback: first entry that doesn't already have geo
    for i, e in enumerate(entries):
        if not _already_has_geo(e["text"]):
            return i, strategy[0]

    return -1, strategy[0]   # -1 means no suitable entry found


def inject_geo_into_entry(text: str, phrase_type: str, rng: random.Random) -> str:
    """Inject a geographic phrase into an existing transcript entry."""
    phrase = rng.choice(GEO_PHRASES[phrase_type])

    # If the entry ends with "Half Moon Bay." — inject before that
    if text.rstrip().endswith("Half Moon Bay."):
        base = text.rstrip()[:-len("Half Moon Bay.")].rstrip().rstrip(",")
        return f"{base}, {phrase}, Half Moon Bay."

    # If it ends with a period, append as a new clause
    if text.rstrip().endswith("."):
        return f"{text.rstrip()[:-1]}, {phrase}."

    # Otherwise just append
    return f"{text.rstrip()}, {phrase}."


def inject_geo_into_advisory(advisory: str, rng: random.Random) -> str:
    """Add KHAF identifier and a local context note to the advisory if not already present."""
    if _already_has_geo(advisory) or "KHAF" in advisory:
        return advisory

    local_note = rng.choice(GEO_PHRASES["advisory_local"])

    # Append as a context note at the end
    if advisory.rstrip().endswith("."):
        return f"{advisory.rstrip()} [{local_note}]"
    return f"{advisory.rstrip()}. [{local_note}]"


def fix_metar(scenario: dict, report: list) -> dict:
    sid = scenario["scenario_id"]
    if sid not in IMC_REPLACEMENT:
        return scenario

    old_metar = scenario["metar"]["raw"]
    scenario["metar"] = copy.deepcopy(IMC_REPLACEMENT[sid])
    report.append(
        f"[FIX-1] {sid} ({scenario['hazard_type']}): METAR patched\n"
        f"         OLD: {old_metar}\n"
        f"         NEW: {scenario['metar']['raw']}"
    )
    return scenario


def fix_geography(scenario: dict, report: list, rng: random.Random) -> dict:
    ht      = scenario["hazard_type"]
    sid     = scenario["scenario_id"]
    entries = scenario["transcript_ground_truth"]
    strategy = HAZARD_PHRASE_STRATEGY.get(ht, ["inbound", "pattern"])

    injected_tx  = False
    injected_adv = False

    idx, phrase_type = _find_injection_entry(entries, strategy)
    if idx >= 0:
        old_text = entries[idx]["text"]
        new_text = inject_geo_into_entry(old_text, phrase_type, rng)
        if new_text != old_text:
            scenario["transcript_ground_truth"][idx]["text"] = new_text
            # Also update raw SRT
            scenario["transcript_ground_truth_raw"] = _rebuild_srt_raw(
                scenario["transcript_ground_truth"]
            )
            report.append(
                f"[FIX-2] {sid} tx[{idx+1}] +{phrase_type}:\n"
                f"         OLD: {old_text[:90]}\n"
                f"         NEW: {new_text[:90]}"
            )
            injected_tx = True

    old_adv = scenario["ground_truth_advisory"]
    new_adv = inject_geo_into_advisory(old_adv, rng)
    if new_adv != old_adv:
        scenario["ground_truth_advisory"] = new_adv
        report.append(
            f"[FIX-2] {sid} advisory +local_context:\n"
            f"         OLD: {old_adv[:90]}\n"
            f"         NEW: {new_adv[:90]}"
        )
        injected_adv = True

    if not injected_tx and not injected_adv:
        report.append(f"[FIX-2] {sid}: already has geo context — skipped")

    return scenario


def _rebuild_srt_raw(entries: list) -> str:
    """Rebuild the raw SRT string from the parsed entries list."""
    lines = []
    for e in entries:
        lines.append(str(e["index"]))
        lines.append(e["timestamp"])
        lines.append(e["text"])
        lines.append("")
    return "\n".join(lines).rstrip()


def validate(original: list, patched: list, report: list):
    report.append("\n=== VALIDATION ===")

    # Fix 1: check imc scenarios now have ceiling
    imc_ok = all(
        s["metar"]["ceiling_ft"] is not None
        for s in patched
        if s["hazard_type"] == "imc_vfr_conflict"
    )
    report.append(f"[VAL] imc_vfr_conflict METAR consistency: {'PASS' if imc_ok else 'FAIL'}")

    # Fix 2: check geo coverage improved
    def has_geo(s):
        full = " ".join(e["text"] for e in s["transcript_ground_truth"])
        return _already_has_geo(full) or "KHAF" in s["ground_truth_advisory"]

    orig_geo  = sum(1 for s in original if has_geo(s))
    patch_geo = sum(1 for s in patched  if has_geo(s))
    report.append(f"[VAL] Geo coverage: {orig_geo}/100 → {patch_geo}/100 scenarios")

    # Label counts unchanged
    orig_labels  = {l: sum(1 for s in original if s["label"] == l) for l in ["nominal","warning","hazard"]}
    patch_labels = {l: sum(1 for s in patched  if s["label"] == l) for l in ["nominal","warning","hazard"]}
    labels_ok = orig_labels == patch_labels
    report.append(f"[VAL] Label distribution preserved: {'PASS' if labels_ok else 'FAIL'}")
    report.append(f"      {orig_labels} → {patch_labels}")

    # Scenario count unchanged
    count_ok = len(original) == len(patched)
    report.append(f"[VAL] Scenario count preserved: {'PASS' if count_ok else 'FAIL'} ({len(patched)})")

    # No empty transcripts created
    empty_tx = [s["scenario_id"] for s in patched if len(s["transcript_ground_truth"]) == 0]
    report.append(f"[VAL] Empty transcripts: {'PASS' if not empty_tx else 'FAIL — ' + str(empty_tx)}")

    # Spot check: S019 METAR should now be IMC
    s019 = next((s for s in patched if s["scenario_id"] == "S019"), None)
    if s019:
        s019_ok = s019["metar"]["ceiling_ft"] == 400
        report.append(f"[VAL] S019 ceiling_ft == 400: {'PASS' if s019_ok else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(
        description="Postprocess CTAF-KHAF dataset: fix METAR mismatches and inject geographic context"
    )
    parser.add_argument("--input",   required=True,  help="Input JSON path")
    parser.add_argument("--output",  default="ctaf_khaf_synthetic_v2.json", help="Output JSON path")
    parser.add_argument("--seed",    type=int, default=42, help="RNG seed for phrase selection")
    parser.add_argument("--dry-run", action="store_true", help="Print changes only, do not write")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading {args.input}...")
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    scenarios_orig   = data["scenarios"]
    scenarios_patched = copy.deepcopy(scenarios_orig)

    report = [
        f"CTAF-KHAF Dataset Postprocessor Report",
        f"Generated: {datetime.now().isoformat()}",
        f"Input:     {args.input}",
        f"Output:    {args.output}",
        f"Scenarios: {len(scenarios_patched)}",
        "=" * 60,
        "",
        "=== FIX 1: METAR MISMATCHES ===",
    ]

    for i, s in enumerate(scenarios_patched):
        scenarios_patched[i] = fix_metar(s, report)

    report.append("")
    report.append("=== FIX 2: KHAF GEOGRAPHIC CONTEXT INJECTION ===")

    for i, s in enumerate(scenarios_patched):
        scenarios_patched[i] = fix_geography(s, report, rng)

    validate(scenarios_orig, scenarios_patched, report)

    report_text = "\n".join(report)
    print(report_text)

    if not args.dry_run:
        # Update dataset metadata
        data["scenarios"] = scenarios_patched
        data["metadata"]["version"]          = "v2"
        data["metadata"]["postprocessed"]    = True
        data["metadata"]["postprocess_fixes"] = [
            "fix_1_metar_mismatches: Replaced clear_vfr METAR with low_imc for imc_vfr_conflict scenarios S019 and S023",
            "fix_2_geo_context: Injected KHAF-specific geographic phrases into transcripts and advisories across all 100 scenarios",
        ]

        # Recompute geo coverage stat for metadata
        def has_geo(s):
            full = " ".join(e["text"] for e in s["transcript_ground_truth"])
            markers = ["pillar point","princeton","montara","el granada","moss beach",
                       "norcal","127.275","135.1","awos","coastline","offshore",
                       "marine layer","coastal fog","ocean","over the water","khaf"]
            return any(m in full.lower() or m in s["ground_truth_advisory"].lower() for m in markers)

        geo_count = sum(1 for s in scenarios_patched if has_geo(s))
        data["metadata"]["geo_context_coverage"] = f"{geo_count}/100"

        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved patched dataset → {out_path}")

        # Write report file
        report_path = out_path.parent / "postprocess_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Saved report         → {report_path}")

    else:
        print("\n[DRY RUN] No files written.")


if __name__ == "__main__":
    main()