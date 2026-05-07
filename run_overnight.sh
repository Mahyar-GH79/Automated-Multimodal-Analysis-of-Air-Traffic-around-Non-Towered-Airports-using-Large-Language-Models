#!/usr/bin/env bash
# ============================================================================
# Overnight runner: GPT-4o CoT + all 3 ablations × {GPT-4o, GPT-5.4} + figures.
#
# Prereq: export OPENAI_API_KEY=sk-...      before running.
# Usage:  bash run_overnight.sh
#         bash run_overnight.sh --dry-run    # print plan, don't execute
#
# Total wallclock estimate: ~3-6 hours (mostly OpenAI API latency, sequential).
# Cost estimate (paid API calls only):
#   GPT-4o   CoT main:                 282 calls × $0.0025/$0.010   ~= $1
#   GPT-4o   whisper ablation: 3 sizes × 3 strats × 94 = 846 calls   ~= $2-3
#   GPT-4o   noise   ablation: 5 levels × 1 strat × 94 = 470 calls   ~= $1-2
#   GPT-4o   mask    ablation: 2 types × 5 rates × 94 = 940 calls    ~= $2-3
#   GPT-5.4  whisper ablation: same shape, ~3-4x  cost              ~= $7-10
#   GPT-5.4  noise   ablation:                                       ~= $4-5
#   GPT-5.4  mask    ablation:                                       ~= $7-10
# Total estimated cost:                                              ~$25-35
# ============================================================================

set -e
set -o pipefail
cd "$(dirname "$0")"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

# ── Sanity ──────────────────────────────────────────────────────────────────
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set. export OPENAI_API_KEY=sk-... and re-run."
    exit 1
fi
if [[ ! -f dataset/ctaf_khaf_synthetic_v2.json ]]; then
    echo "ERROR: dataset/ctaf_khaf_synthetic_v2.json not found."
    exit 1
fi
if [[ ! -f results/metrics/all_metrics.json ]]; then
    echo "ERROR: results/metrics/all_metrics.json missing. Run make_figures.py first."
    exit 1
fi

DATASET=dataset/ctaf_khaf_synthetic_v2.json
AUDIO_DIR=dataset/scenarios
GT_METRICS=results/metrics/all_metrics.json

LOG_DIR=results/run_overnight_logs
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_overnight.log"

run() {
    local desc=$1; shift
    echo ""
    echo "============================================================"
    echo "  [$(date +%H:%M:%S)] $desc"
    echo "  CMD: $*"
    echo "============================================================"
    if [[ $DRY -eq 1 ]]; then
        echo "  (dry-run, skipped)"
        return 0
    fi
    "$@" 2>&1 | tee -a "$RUN_LOG"
}

echo "==================================================================="
echo "  CTAF-KHAF Overnight Runner"
echo "  Started: $(date)"
echo "  Log: $RUN_LOG"
echo "  Dry-run: $DRY"
echo "==================================================================="

# ── 1. GPT-4o CoT (3 strategies) ─────────────────────────────────────────────
# Currently GPT-4o has only direct-prompting runs in results/. This adds CoT.
# Existing direct runs are NOT touched (different run_keys).
run "Step 1/5: GPT-4o CoT (zero/one/few-shot)" \
    python run_experiments.py \
        --dataset "$DATASET" \
        --models gpt-4o \
        --strategies zero_shot one_shot few_shot \
        --cot \
        --out results

# ── 2. Whisper ASR ablation: GPT-4o + GPT-5.4 ────────────────────────────────
# Transcripts already exist from the previous open-source run, so we skip ASR.
run "Step 2/5: Whisper ablation (GPT-4o + GPT-5.4)" \
    python ablation_whisper.py \
        --dataset    "$DATASET" \
        --audio-dir  "$AUDIO_DIR" \
        --gt-metrics "$GT_METRICS" \
        --out        results/ablation_whisper \
        --models     gpt-4o gpt-5.4 \
        --skip-transcription

# ── 3. Audio-noise ablation: GPT-4o + GPT-5.4 ────────────────────────────────
# Noisy audio + Whisper transcripts also already cached.
run "Step 3/5: Noise ablation (GPT-4o + GPT-5.4)" \
    python ablation_audio_noise.py \
        --dataset    "$DATASET" \
        --audio-dir  "$AUDIO_DIR" \
        --gt-metrics "$GT_METRICS" \
        --out        results/ablation_noise \
        --models     gpt-4o gpt-5.4

# ── 4. Text-mask ablation: GPT-4o + GPT-5.4 ──────────────────────────────────
run "Step 4/5: Mask ablation (GPT-4o + GPT-5.4)" \
    python ablation_text_mask.py \
        --dataset    "$DATASET" \
        --gt-metrics "$GT_METRICS" \
        --out        results/ablation_mask \
        --models     gpt-4o gpt-5.4

# ── 5. Regenerate figures (main + ablations) ─────────────────────────────────
run "Step 5/5a: Main figures (rebuild metrics + plot)" \
    python make_figures.py

run "Step 5/5b: Whisper-ablation figures (plots-only on enriched metrics)" \
    python ablation_whisper.py \
        --dataset    "$DATASET" \
        --audio-dir  "$AUDIO_DIR" \
        --gt-metrics "$GT_METRICS" \
        --out        results/ablation_whisper \
        --plots-only

run "Step 5/5c: Noise-ablation figures" \
    python ablation_audio_noise.py \
        --dataset    "$DATASET" \
        --audio-dir  "$AUDIO_DIR" \
        --gt-metrics "$GT_METRICS" \
        --out        results/ablation_noise \
        --plots-only

run "Step 5/5d: Mask-ablation figures" \
    python ablation_text_mask.py \
        --dataset    "$DATASET" \
        --gt-metrics "$GT_METRICS" \
        --out        results/ablation_mask \
        --plots-only

echo ""
echo "==================================================================="
echo "  All steps complete. Finished: $(date)"
echo "  Full log: $RUN_LOG"
echo "==================================================================="
