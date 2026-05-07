#!/usr/bin/env bash
# ============================================================================
# Overnight runner: add Claude Sonnet 4.6 to 3-class, full binary benchmark
# (6 models x 3 strategies x 2 CoT) on all 6 models, then regenerate figures
# (3-class + binary, including PR/ROC for binary).
#
# Prereqs (env vars):
#   export OPENAI_API_KEY=sk-...
#   export ANTHROPIC_API_KEY=sk-ant-...
#
# Usage:
#   bash run_overnight_binary.sh
#   bash run_overnight_binary.sh --dry-run
#
# Cost estimate (paid API calls only):
#   Claude on 3-class:        6 runs x 94 scen x ~$0.005     ~$3
#   Claude on binary:         6 runs x 94 scen x ~$0.005     ~$3
#   GPT-4o on binary:         6 runs x 94 scen x ~$0.0035    ~$2
#   GPT-5.4 on binary:        6 runs x 94 scen x ~$0.012     ~$7
#   OpenAI logprob scoring:   adds 1 short call per scenario  ~$2
# Total estimated:                                            ~$15-20
#
# Wallclock estimate: 4-7 hours (sequential, dominated by open-source GPU
# inference for binary + logprob scoring pass on every record).
# ============================================================================

set -e
set -o pipefail
cd "$(dirname "$0")"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

# ── Sanity ──────────────────────────────────────────────────────────────────
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set."
    exit 1
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY not set."
    exit 1
fi
if [[ ! -f dataset/ctaf_khaf_synthetic_v2.json ]]; then
    echo "ERROR: dataset/ctaf_khaf_synthetic_v2.json not found."
    exit 1
fi

DATASET=dataset/ctaf_khaf_synthetic_v2.json

LOG_DIR=results/run_overnight_logs
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_binary.log"

run() {
    local desc=$1; shift
    echo ""
    echo "=========================================================="
    echo "  [$(date +%H:%M:%S)] $desc"
    echo "  CMD: $*"
    echo "=========================================================="
    if [[ $DRY -eq 1 ]]; then
        echo "  (dry-run, skipped)"
        return 0
    fi
    "$@" 2>&1 | tee -a "$RUN_LOG"
}

echo "=========================================================="
echo "  CTAF-KHAF Overnight Runner (Binary + Claude)"
echo "  Started: $(date)"
echo "  Log: $RUN_LOG"
echo "  Dry-run: $DRY"
echo "=========================================================="

# ── Step 1: Claude Sonnet 4.6 on the existing 3-class benchmark ─────────────
# Adds claude_*_(zero|one|few)_shot[_cot] runs to results/raw/. Resume-safe;
# existing OpenAI/open-source runs are untouched.
run "1/8: Claude Sonnet 4.6 on 3-class, direct prompting" \
    python run_experiments.py \
        --dataset "$DATASET" \
        --closed-source-only \
        --claude-model claude-sonnet-4-6 \
        --strategies zero_shot one_shot few_shot

run "2/8: Claude Sonnet 4.6 on 3-class, CoT" \
    python run_experiments.py \
        --dataset "$DATASET" \
        --closed-source-only \
        --claude-model claude-sonnet-4-6 \
        --strategies zero_shot one_shot few_shot \
        --cot

# ── Step 2: Binary benchmark on open-source models ──────────────────────────
# Runs each (qwen, mistral, gemma) for direct + CoT, with logprob scoring on
# every scenario. Output: results_binary/raw/.
run "3/8: Open-source models on binary, direct" \
    python run_experiments.py \
        --dataset "$DATASET" \
        --binary \
        --models qwen mistral gemma \
        --strategies zero_shot one_shot few_shot

run "4/8: Open-source models on binary, CoT" \
    python run_experiments.py \
        --dataset "$DATASET" \
        --binary \
        --models qwen mistral gemma \
        --strategies zero_shot one_shot few_shot \
        --cot

# ── Step 3: Binary benchmark on closed-source models ────────────────────────
run "5/8: GPT-4o on binary, direct" \
    python run_experiments.py \
        --dataset "$DATASET" \
        --binary --closed-source-only \
        --closed-source-model gpt-4o \
        --strategies zero_shot one_shot few_shot

run "5b/8: GPT-4o on binary, CoT" \
    python run_experiments.py \
        --dataset "$DATASET" \
        --binary --closed-source-only \
        --closed-source-model gpt-4o \
        --strategies zero_shot one_shot few_shot \
        --cot

run "6/8: GPT-5.4 on binary, direct" \
    python run_experiments.py \
        --dataset "$DATASET" \
        --binary --closed-source-only \
        --closed-source-model gpt-5.4 \
        --strategies zero_shot one_shot few_shot

run "6b/8: GPT-5.4 on binary, CoT" \
    python run_experiments.py \
        --dataset "$DATASET" \
        --binary --closed-source-only \
        --closed-source-model gpt-5.4 \
        --strategies zero_shot one_shot few_shot \
        --cot

run "7/8: Claude Sonnet 4.6 on binary, direct" \
    python run_experiments.py \
        --dataset "$DATASET" \
        --binary --closed-source-only \
        --claude-model claude-sonnet-4-6 \
        --strategies zero_shot one_shot few_shot

run "7b/8: Claude Sonnet 4.6 on binary, CoT" \
    python run_experiments.py \
        --dataset "$DATASET" \
        --binary --closed-source-only \
        --claude-model claude-sonnet-4-6 \
        --strategies zero_shot one_shot few_shot \
        --cot

# ── Step 4: Regenerate all figures ──────────────────────────────────────────
run "8a/8: Rebuild 3-class figures (auto-detects 3-class)" \
    python make_figures.py --raw-dir results/raw --out results

run "8b/8: Rebuild binary figures (auto-detects binary, includes PR/ROC)" \
    python make_figures.py --raw-dir results_binary/raw --out results_binary

echo ""
echo "=========================================================="
echo "  All steps complete. Finished: $(date)"
echo "  Log: $RUN_LOG"
echo "=========================================================="
echo ""
echo "  Outputs:"
echo "    3-class : results/figures/, results/tables/"
echo "    Binary  : results_binary/figures/ (+ pr_roc_curves.pdf), results_binary/tables/"
