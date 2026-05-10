#!/bin/bash
#SBATCH --job-name=scidiscover-sanity
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=/cta/users/sait.kacmaz/logs/scidiscover-sanity-%j.out
#SBATCH --error=/cta/users/sait.kacmaz/logs/scidiscover-sanity-%j.err

PROJECT=/cta/users/sait.kacmaz/workfolder/scidiscovertest
cd "$PROJECT" || exit 1

mkdir -p /cta/users/sait.kacmaz/logs logs reports

echo "=== Sanity job started at $(date) ==="
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
# Run 1: Full pipeline (reranker + verifier)
# -----------------------------------------------------------
echo ""
echo "=== [1/4] Full pipeline (reranker + verifier) — $(date) ==="
PYTHONPATH=. .venv/bin/python -m scidiscover.eval.run \
    --config configs/demo.yaml \
    --max-queries 3 \
    --output reports/sanity_full.json
echo "=== [1/4] Done — $(date) ==="

# -----------------------------------------------------------
# Run 2: No reranker
# -----------------------------------------------------------
echo ""
echo "=== [2/4] No reranker — $(date) ==="
PYTHONPATH=. .venv/bin/python -m scidiscover.eval.run \
    --config configs/demo.yaml \
    --skip-reranker \
    --max-queries 1 \
    --output reports/sanity_no_reranker.json
echo "=== [2/4] Done — $(date) ==="

# -----------------------------------------------------------
# Run 3: No verifier
# -----------------------------------------------------------
echo ""
echo "=== [3/4] No verifier — $(date) ==="
PYTHONPATH=. .venv/bin/python -m scidiscover.eval.run \
    --config configs/demo.yaml \
    --skip-verifier \
    --max-queries 1 \
    --output reports/sanity_no_verifier.json
echo "=== [3/4] Done — $(date) ==="

# -----------------------------------------------------------
# Run 4: Bare (no reranker, no verifier)
# -----------------------------------------------------------
echo ""
echo "=== [4/4] Bare (no reranker, no verifier) — $(date) ==="
PYTHONPATH=. .venv/bin/python -m scidiscover.eval.run \
    --config configs/demo.yaml \
    --skip-reranker \
    --skip-verifier \
    --max-queries 1 \
    --output reports/sanity_bare.json
echo "=== [4/4] Done — $(date) ==="

echo ""
echo "=== Sanity job finished at $(date) ==="
