#!/bin/bash
#SBATCH --job-name=scidiscover-qwen-no-critique
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-03:00:00
#SBATCH --output=/cta/users/sait.kacmaz/logs/scidiscover-qwen-no-critique-%j.out
#SBATCH --error=/cta/users/sait.kacmaz/logs/scidiscover-qwen-no-critique-%j.err

PROJECT=/cta/users/sait.kacmaz/workfolder/scidiscovertest
cd "$PROJECT" || exit 1

mkdir -p /cta/users/sait.kacmaz/logs logs/critique_loop_false outputs/critique_loop_false reports

echo "=== Qwen no-critique eval job started at $(date) ==="
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
echo "=== Qwen run (critique_loop=false, reranker=cross-encoder) — $(date) ==="
PYTHONPATH=. .venv/bin/python -m scidiscover.eval.run \
    --config configs/critique_loop_false.yaml \
    --output "reports/qwen_no_critique_run_1.json"
echo "=== Run done — $(date) ==="

echo ""
echo "=== Qwen no-critique eval job finished at $(date) ==="
