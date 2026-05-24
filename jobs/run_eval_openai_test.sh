#!/bin/bash
#SBATCH --job-name=scidiscover-openai-test
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --array=1-2
#SBATCH --output=/cta/users/sait.kacmaz/logs/scidiscover-openai-test-%A_%a.out
#SBATCH --error=/cta/users/sait.kacmaz/logs/scidiscover-openai-test-%A_%a.err

PROJECT=/cta/users/sait.kacmaz/workfolder/scidiscovertest
cd "$PROJECT" || exit 1

mkdir -p /cta/users/sait.kacmaz/logs logs/openai outputs/openai reports

echo "=== OpenAI TEST job started (run ${SLURM_ARRAY_TASK_ID}/2) at $(date) ==="
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

echo ""
echo "=== OpenAI test run ${SLURM_ARRAY_TASK_ID} (2 queries) — $(date) ==="
PYTHONPATH=. .venv/bin/python -m scidiscover.eval.run \
    --config configs/openai.yaml \
    --max-queries 2 \
    --output "reports/openai_test_run_${SLURM_ARRAY_TASK_ID}.json"
echo "=== Done — $(date) ==="

echo ""
echo "=== OpenAI TEST job finished (run ${SLURM_ARRAY_TASK_ID}/2) at $(date) ==="
