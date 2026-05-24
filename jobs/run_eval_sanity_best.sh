#!/bin/bash
#SBATCH --job-name=scidiscover-sanity-best
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=/cta/users/sait.kacmaz/logs/scidiscover-sanity-best-%j.out
#SBATCH --error=/cta/users/sait.kacmaz/logs/scidiscover-sanity-best-%j.err

PROJECT=/cta/users/sait.kacmaz/workfolder/scidiscovertest
cd "$PROJECT" || exit 1

mkdir -p /cta/users/sait.kacmaz/logs logs reports

echo "=== Best-config sanity job started at $(date) ==="
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
# (critique_loop.enabled: true set in configs/demo.yaml)
# -----------------------------------------------------------
echo ""
echo "=== Best-config sanity run (reranker + verifier + critique loop, 3 queries) — $(date) ==="
PYTHONPATH=. .venv/bin/python -m scidiscover.eval.run \
    --config configs/demo.yaml \
    --max-queries 3 \
    --output reports/sanity_best.json
echo "=== Done — $(date) ==="

echo ""
echo "=== Best-config sanity job finished at $(date) ==="
