#!/bin/bash
#SBATCH --job-name=ultimate_scidiscover
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-03:00:00
#SBATCH --output=/cta/users/sait.kacmaz/logs/scidiscover-ultimate-%j.out
#SBATCH --error=/cta/users/sait.kacmaz/logs/scidiscover-ultimate-%j.err

PROJECT=/cta/users/sait.kacmaz/workfolder/scidiscovertest
cd "$PROJECT" || exit 1

mkdir -p /cta/users/sait.kacmaz/logs logs/openai outputs/openai reports

echo "=== ultimate_scidiscover eval job started at $(date) ==="
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
echo "=== ultimate_scidiscover run (retrieval-k=20, reranker + verifier + critique loop) — $(date) ==="
PYTHONPATH=. .venv/bin/python -m scidiscover.eval.run \
    --config configs/openai.yaml \
    --retrieval-k 20 \
    --output "reports/ultimate_scidiscover_run_1.json"
echo "=== Run done — $(date) ==="

echo ""
echo "=== ultimate_scidiscover eval job finished at $(date) ==="
