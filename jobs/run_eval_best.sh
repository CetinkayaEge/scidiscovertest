#!/bin/bash
#SBATCH --job-name=scidiscover-eval-qwen
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-3
#SBATCH --output=/cta/users/sait.kacmaz/logs/scidiscover-eval-qwen-%A_%a.out
#SBATCH --error=/cta/users/sait.kacmaz/logs/scidiscover-eval-qwen-%A_%a.err

PROJECT=/cta/users/sait.kacmaz/workfolder/scidiscovertest
cd "$PROJECT" || exit 1

mkdir -p /cta/users/sait.kacmaz/logs logs reports

echo "=== Qwen eval job started (run ${SLURM_ARRAY_TASK_ID}/3) at $(date) ==="
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
# Model: local Qwen 2.5 32B (configs/demo.yaml)
# Each array task writes to its own output file.
# -----------------------------------------------------------
echo ""
echo "=== Qwen full run ${SLURM_ARRAY_TASK_ID} (reranker + verifier + critique loop) — $(date) ==="
PYTHONPATH=. .venv/bin/python -m scidiscover.eval.run \
    --config configs/demo.yaml \
    --output "reports/qwen_run_${SLURM_ARRAY_TASK_ID}.json"
echo "=== Done — $(date) ==="

echo ""
echo "=== Qwen eval job finished (run ${SLURM_ARRAY_TASK_ID}/3) at $(date) ==="
