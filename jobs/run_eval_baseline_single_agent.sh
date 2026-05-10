#!/bin/bash
#SBATCH --job-name=scidiscover-baseline
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/cta/users/sait.kacmaz/logs/scidiscover-baseline-%j.out
#SBATCH --error=/cta/users/sait.kacmaz/logs/scidiscover-baseline-%j.err

PROJECT=/cta/users/sait.kacmaz/workfolder/scidiscovertest
cd "$PROJECT" || exit 1

mkdir -p /cta/users/sait.kacmaz/logs logs reports

echo "=== Baseline single-agent job started at $(date) ==="
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
# Baseline: single agent
# -----------------------------------------------------------
echo ""
echo "=== Baseline single-agent run — $(date) ==="
PYTHONPATH=. .venv/bin/python -m scidiscover.eval.run \
    --config configs/demo.yaml \
    --baseline single_agent \
    --output reports/eval_baseline_single_agent.json
echo "=== Done — $(date) ==="

echo ""
echo "=== Job finished at $(date) ==="
