#!/bin/bash
# ============================================================
# SLURM job: PaperQA2 SOTA baseline + comparison table
# ============================================================
# Resources: no GPU needed (API-based run).
# Using the same partition/qos as the main eval jobs for consistency.
# Adjust --partition / --qos if your cluster has a CPU-only queue.
#
# Estimated runtime:
#   - First run:  ~20–40 min (OpenAI embedding for ~22 K papers + 40 queries)
#   - Repeat run: ~10–15 min (index cached; only 40 API calls)
#
# Submit with:
#   sbatch paperqa_baseline/run_paperqa_hpc.sh
# ============================================================
#SBATCH --job-name=scidiscover-paperqa-baseline
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-02:00:00
#SBATCH --output=/cta/users/sait.kacmaz/logs/scidiscover-paperqa-%j.out
#SBATCH --error=/cta/users/sait.kacmaz/logs/scidiscover-paperqa-%j.err

PROJECT=/cta/users/sait.kacmaz/workfolder/scidiscovertest
cd "$PROJECT" || exit 1

mkdir -p /cta/users/sait.kacmaz/logs paperqa_baseline/results

echo "=== PaperQA2 baseline job started at $(date) ==="
echo "=== Running on node: $(hostname) ==="

# -----------------------------------------------------------
# Load environment variables from .env
# OPENAI_API_KEY_SOTA must be present in .env (never hard-code it here).
# OPENAI_BASE_URL and OPENAI_API_KEY are the HPC vLLM settings used by
# the main pipeline; env_utils.py temporarily overrides them for PaperQA2.
# -----------------------------------------------------------
set -a
source .env
set +a

# Safety check: abort if the SOTA key is missing
if [ -z "$OPENAI_API_KEY_SOTA" ] || [ "$OPENAI_API_KEY_SOTA" = "sk-..." ]; then
    echo "ERROR: OPENAI_API_KEY_SOTA is not set in .env — aborting."
    echo "       Add your real OpenAI API key as OPENAI_API_KEY_SOTA=sk-<your-key>"
    exit 1
fi

# -----------------------------------------------------------
# Setup venv (reuse if already present)
# -----------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "=== Creating virtual environment ==="
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

# Install paper-qa if not already present
if ! .venv/bin/python -c "import paperqa" 2>/dev/null; then
    echo "=== Installing paper-qa ==="
    .venv/bin/pip install paper-qa
fi

# -----------------------------------------------------------
# Step 1: Run PaperQA2 baseline (all 40 queries, gpt-4o-mini)
# -----------------------------------------------------------
echo ""
echo "=== [1/2] PaperQA2 baseline run — $(date) ==="

PYTHONPATH=. .venv/bin/python paperqa_baseline/run_paperqa.py \
    --model gpt-4o-mini \
    --output paperqa_baseline/results/paperqa_results.json

STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "ERROR: run_paperqa.py exited with status $STATUS — skipping comparison."
    exit $STATUS
fi
echo "=== [1/2] Done — $(date) ==="

# -----------------------------------------------------------
# Step 2: Generate comparison table
# -----------------------------------------------------------
echo ""
echo "=== [2/2] Generating comparison table — $(date) ==="

# Point at the most recent full eval report if it exists; otherwise the
# compare script will gracefully omit the SciDiscover column.
OUR_RESULTS="reports/eval_results.json"
if [ ! -f "$OUR_RESULTS" ]; then
    # Fall back to whichever full eval report exists
    OUR_RESULTS=$(ls reports/eval_full*.json 2>/dev/null | tail -1)
fi

if [ -n "$OUR_RESULTS" ] && [ -f "$OUR_RESULTS" ]; then
    PYTHONPATH=. .venv/bin/python paperqa_baseline/compare_results.py \
        --our-results "$OUR_RESULTS" \
        --pqa-results paperqa_baseline/results/paperqa_results.json \
        --output paperqa_baseline/results/comparison_table.md
else
    PYTHONPATH=. .venv/bin/python paperqa_baseline/compare_results.py \
        --pqa-results paperqa_baseline/results/paperqa_results.json \
        --output paperqa_baseline/results/comparison_table.md
fi

echo "=== [2/2] Done — $(date) ==="

echo ""
echo "=== PaperQA2 baseline job finished at $(date) ==="
echo "=== Results: paperqa_baseline/results/paperqa_results.json ==="
echo "=== Table:   paperqa_baseline/results/comparison_table.md ==="
