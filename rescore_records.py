#!/usr/bin/env python3
"""Patch raw prediction JSONs with confidence-derived class scores when logprobs are missing or uniform."""

import json
import sys
from pathlib import Path


def is_uniform_or_empty(scores: dict) -> bool:
    if not scores:
        return True
    vals = list(scores.values())
    if len(vals) < 2:
        return True
    # Uniform within float epsilon
    return max(vals) - min(vals) < 1e-9


def confidence_score(predicted: str, confidence: float, classes: list) -> dict:
    if predicted in classes and 0.0 <= confidence <= 1.0:
        other = (1.0 - confidence) / max(len(classes) - 1, 1)
        return {l: (confidence if l == predicted else other) for l in classes}
    return {l: 1.0 / len(classes) for l in classes}


def patch_file(path: Path) -> tuple[int, int]:
    """Returns (n_records_patched, n_records_total)."""
    data = json.load(open(path))
    records = data.get("records", [])
    if not records:
        return 0, 0

    classes = sorted({r.get("ground_truth") for r in records
                      if r.get("ground_truth")} |
                     {r.get("predicted")    for r in records
                      if r.get("predicted") in
                         {"nominal", "warning", "hazard", "danger"}})
    # Stable ordering (nominal first if present)
    ordered = [c for c in ("nominal", "warning", "hazard", "danger")
               if c in classes]
    if not ordered:
        return 0, len(records)

    n_patched = 0
    for r in records:
        cs = r.get("class_scores") or {}
        if is_uniform_or_empty(cs) or set(cs.keys()) != set(ordered):
            r["class_scores"] = confidence_score(
                r.get("predicted", ""),
                float(r.get("confidence", 0.5)),
                ordered)
            r["score_source"] = "confidence_fallback"
            n_patched += 1

    if n_patched:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    return n_patched, len(records)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python rescore_records.py <raw_dir> [<raw_dir2> ...]")

    total_files = total_patched = total_records = 0
    for arg in sys.argv[1:]:
        raw_dir = Path(arg)
        if not raw_dir.is_dir():
            print(f"  [skip] {raw_dir} not a directory")
            continue
        print(f"Scanning {raw_dir}/")
        for f in sorted(raw_dir.glob("*.json")):
            n_pat, n_total = patch_file(f)
            if n_pat:
                print(f"  patched {n_pat:3d}/{n_total} records in {f.name}")
                total_patched += n_pat
            total_files += 1
            total_records += n_total
    print(f"\nDone. {total_patched} records patched across {total_files} files "
          f"({total_records} records scanned).")


if __name__ == "__main__":
    main()
