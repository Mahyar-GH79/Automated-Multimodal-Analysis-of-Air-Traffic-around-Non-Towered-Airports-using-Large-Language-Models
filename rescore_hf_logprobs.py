#!/usr/bin/env python3
"""rescore_hf_logprobs.py"""

import argparse
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

# Reuse the existing run_experiments machinery (system prompts, ICL pool,
# build_messages, load_model, parse_run_key, apply_binary_mode, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as RE


def score_classes_robust(model, tokenizer, messages, classes):
    """First-token logprob scoring with proper chat-template handling."""
    import torch

    # Append the constraint to the last user message (preserves alternation)
    prompt_msgs = [dict(m) for m in messages]
    instruction = ("\n\nOutput exactly one word and nothing else, chosen "
                   "from this list: " + ", ".join(classes) + ".")
    if prompt_msgs and prompt_msgs[-1].get("role") == "user":
        prompt_msgs[-1] = {
            "role":    "user",
            "content": prompt_msgs[-1]["content"] + instruction,
        }
    else:
        prompt_msgs.append({"role": "user", "content": instruction.lstrip()})

    chat_out = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True,
        return_tensors="pt")
    if hasattr(chat_out, "input_ids"):
        ids = chat_out.input_ids.to(model.device)
    elif isinstance(chat_out, dict):
        ids = chat_out["input_ids"].to(model.device)
    else:
        ids = chat_out.to(model.device)

    # Single forward pass. Cast logits to fp32 before log-softmax for
    # numerical stability on models that load in fp16.
    with torch.no_grad():
        next_token_logits = model(ids).logits[0, -1].float()

    cand_logits = []
    for c in classes:
        best = -1e9
        for variant in (" " + c, c, " " + c.capitalize(), c.capitalize()):
            tok_ids = tokenizer.encode(variant, add_special_tokens=False)
            if not tok_ids:
                continue
            lg = float(next_token_logits[tok_ids[0]].item())
            if lg > best:
                best = lg
        cand_logits.append(best)

    mx = max(cand_logits)
    exps = [math.exp(l - mx) for l in cand_logits]
    total = sum(exps) or 1.0
    return {c: e / total for c, e in zip(classes, exps)}


def parse_run_key(rk: str, model_key: str):
    """qwen_zero_shot_cot -> ('zero_shot', True)"""
    body = rk[len(model_key) + 1:] if rk.startswith(model_key + "_") else rk
    cot = body.endswith("_cot")
    if cot:
        body = body[:-4]
    return body, cot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",
                    default="dataset/ctaf_khaf_synthetic_v2.json")
    ap.add_argument("--raw-dir",
                    default="results_binary/raw",
                    help="Directory of raw prediction JSONs to update.")
    ap.add_argument("--models", nargs="+",
                    default=["qwen", "mistral", "gemma"],
                    choices=["qwen", "mistral", "gemma"],
                    help="HF models to rescore (default: all three, so "
                         "the open-source group shares one scoring method).")
    ap.add_argument("--quantize", default=None, choices=["4bit", "8bit"])
    ap.add_argument("--backup", action="store_true",
                    help="Back up each raw file to *.bak before overwriting.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N records per file (debugging).")
    args = ap.parse_args()

    # ── Switch run_experiments globals to binary mode (SYSTEM_PROMPT,
    #    LABELS, COT_*, _LABEL_SYNONYMS) so build_messages reconstructs
    #    EXACTLY the prompt the records were generated from. ──
    RE.apply_binary_mode()

    # ── Load dataset, collapse labels, build the same ICL pool the
    #    binary runs used. ──
    print(f"Loading dataset {args.dataset}...")
    with open(args.dataset) as f:
        data = json.load(f)
    all_scenarios = data["scenarios"]
    for s in all_scenarios:
        s["label"] = RE.collapse_label(s["label"])

    icl_ids = set(RE.ICL_EXAMPLE_IDS.values())
    icl_pool = {s["scenario_id"]: s for s in all_scenarios
                if s["scenario_id"] in icl_ids}
    test_set = [s for s in all_scenarios if s["scenario_id"] not in icl_ids]
    icl_ordered = [
        icl_pool[RE.ICL_EXAMPLE_IDS["nominal_1"]],
        icl_pool[RE.ICL_EXAMPLE_IDS["nominal_2"]],
        icl_pool[RE.ICL_EXAMPLE_IDS["warning_1"]],
        icl_pool[RE.ICL_EXAMPLE_IDS["warning_2"]],
        icl_pool[RE.ICL_EXAMPLE_IDS["hazard_1"]],
        icl_pool[RE.ICL_EXAMPLE_IDS["hazard_2"]],
    ]
    sid_to_scenario = {s["scenario_id"]: s for s in test_set}
    print(f"Test set: {len(test_set)}  ICL pool: {len(icl_pool)}  "
          f"LABELS={RE.LABELS}")

    log = logging.getLogger("rescore")
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                                      datefmt="%H:%M:%S"))
    log.addHandler(h)

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        sys.exit(f"raw-dir not found: {raw_dir}")

    import torch

    grand_total = grand_updated = 0
    t_start = time.time()

    for model_key in args.models:
        print(f"\n{'='*68}")
        print(f"Model: {RE.MODELS[model_key]['display']}")
        print(f"{'='*68}")

        files = sorted(raw_dir.glob(f"{model_key}_*.json"))
        if not files:
            print(f"  No files found matching {model_key}_*.json — skipping.")
            continue

        print(f"  Loading {RE.MODELS[model_key]['display']} weights...")
        t0 = time.time()
        model, tokenizer = RE.load_model(model_key, args.quantize, log)
        model.eval()
        print(f"  Loaded in {time.time() - t0:.1f}s on "
              f"{next(model.parameters()).device}")

        for f in files:
            rk = f.stem
            strategy, use_cot = parse_run_key(rk, model_key)
            d = json.load(open(f))
            records = d.get("records", [])
            if args.limit:
                records = records[: args.limit]
            print(f"\n  {f.name}  (strategy={strategy}, "
                  f"cot={use_cot}, n={len(records)})")

            if args.backup:
                bkp = f.with_suffix(".json.bak")
                shutil.copy2(f, bkp)
                print(f"    backed up -> {bkp.name}")

            updated = failed = 0
            failure_reasons = {}
            t_file = time.time()
            for ri, r in enumerate(records):
                sid = r["scenario_id"]
                sc = sid_to_scenario.get(sid)
                if sc is None:
                    failed += 1
                    failure_reasons["scenario_not_found"] = (
                        failure_reasons.get("scenario_not_found", 0) + 1)
                    continue
                try:
                    messages = RE.build_messages(
                        sc, icl_ordered, strategy,
                        model_key=model_key, cot=use_cot)
                    scores = score_classes_robust(
                        model, tokenizer, messages, RE.LABELS)
                    r["class_scores"] = scores
                    r["score_source"] = "logprobs"
                    updated += 1
                except Exception as e:
                    failed += 1
                    key = type(e).__name__
                    failure_reasons[key] = failure_reasons.get(key, 0) + 1
                    # Print first 3 failures of each type so we can see
                    # what's going wrong without spamming the log.
                    if failure_reasons[key] <= 3:
                        print(f"    [{ri+1:3d}/{len(records)}] {sid}: "
                              f"{key}: {str(e)[:160]}")

                if (ri + 1) % 25 == 0:
                    rate = (ri + 1) / (time.time() - t_file)
                    eta = (len(records) - ri - 1) / max(rate, 1e-6)
                    print(f"    [{ri+1:3d}/{len(records)}] "
                          f"{rate:.1f} rec/s  eta {eta:.0f}s")

            with open(f, "w") as out:
                json.dump(d, out, indent=2)
            dt = time.time() - t_file
            summary = (f"    -> {updated} updated, {failed} failed in "
                       f"{dt:.1f}s ({len(records)/max(dt,1e-6):.1f} rec/s)")
            if failure_reasons:
                summary += "    failures: " + ", ".join(
                    f"{k}={v}" for k, v in failure_reasons.items())
            print(summary)
            grand_total += len(records)
            grand_updated += updated

        del model, tokenizer
        torch.cuda.empty_cache()
        print(f"  GPU freed after {RE.MODELS[model_key]['display']}.")

    total_dt = time.time() - t_start
    print(f"\n{'='*68}")
    print(f"All done. {grand_updated}/{grand_total} records re-scored "
          f"in {total_dt:.1f}s.")
    print("Next:")
    print("  python make_paper_assets.py")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
