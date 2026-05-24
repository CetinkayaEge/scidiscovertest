#!/bin/bash
#SBATCH --job-name=scidiscover-eval-openai
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-3
#SBATCH --output=/cta/users/sait.kacmaz/logs/scidiscover-eval-openai-%A_%a.out
#SBATCH --error=/cta/users/sait.kacmaz/logs/scidiscover-eval-openai-%A_%a.err

PROJECT=/cta/users/sait.kacmaz/workfolder/scidiscovertest
cd "$PROJECT" || exit 1

mkdir -p /cta/users/sait.kacmaz/logs logs/openai outputs/openai reports

echo "=== OpenAI eval job started (run ${SLURM_ARRAY_TASK_ID}/3) at $(date) ==="
echo "=== Running on node: $(hostname) ==="

# Load environment variables
set -a
source .env
set +a

# Setup venv only if missing
if [ ! -d ".venv" ]; then
    echo "=== Creating virtual environment ==="
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

# -----------------------------------------------------------
# Best pipeline: reranker on, verifier on, critique loop on
# Model: openrouter-openai/gpt-5.4-mini (configs/openai.yaml)
# Traces go to logs/openai/ to avoid collision with qwen jobs.
# Each array task writes to its own output file.
# -----------------------------------------------------------
echo ""
echo "=== OpenAI full run ${SLURM_ARRAY_TASK_ID} (reranker + verifier + critique loop) — $(date) ==="
PYTHONPATH=. .venv/bin/python -m scidiscover.eval.run \
    --config configs/openai.yaml \
    --output "reports/openai_run_${SLURM_ARRAY_TASK_ID}.json"
echo "=== Done — $(date) ==="

echo ""
echo "=== OpenAI eval job finished (run ${SLURM_ARRAY_TASK_ID}/3) at $(date) ==="
