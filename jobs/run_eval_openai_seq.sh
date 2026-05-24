#!/bin/bash
#SBATCH --job-name=scidiscover-eval-openai-seq
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-08:00:00
#SBATCH --output=/cta/users/sait.kacmaz/logs/scidiscover-eval-openai-seq-%j.out
#SBATCH --error=/cta/users/sait.kacmaz/logs/scidiscover-eval-openai-seq-%j.err

PROJECT=/cta/users/sait.kacmaz/workfolder/scidiscovertest
cd "$PROJECT" || exit 1

mkdir -p /cta/users/sait.kacmaz/logs logs/openai outputs/openai reports

echo "=== OpenAI sequential eval job started at $(date) ==="
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
# Runs 1→2→3 sequentially to avoid OpenRouter rate limiting.
# Traces go to logs/openai/ to avoid collision with qwen jobs.
# -----------------------------------------------------------
for RUN in 1 2 3; do
    echo ""
    echo "=== OpenAI full run ${RUN}/3 (reranker + verifier + critique loop) — $(date) ==="
    PYTHONPATH=. .venv/bin/python -m scidiscover.eval.run \
        --config configs/openai.yaml \
        --output "reports/openai_run_${RUN}.json"
    echo "=== Run ${RUN}/3 done — $(date) ==="
done

echo ""
echo "=== OpenAI sequential eval job finished at $(date) ==="
