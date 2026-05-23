#!/bin/bash
#SBATCH --job-name=scidiscover-ui
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/cta/users/sait.kacmaz/logs/scidiscover-ui-%j.out
#SBATCH --error=/cta/users/sait.kacmaz/logs/scidiscover-ui-%j.err

PROJECT=/cta/users/sait.kacmaz/workfolder/scidiscovertest
cd "$PROJECT" || exit 1

mkdir -p /cta/users/sait.kacmaz/logs logs outputs

echo "=== UI job started at $(date) ==="
echo "=== Running on node: $(hostname) ==="

# Load environment variables
set -a
source .env
set +a

# Create venv if missing, then always sync requirements
if [ ! -d ".venv" ]; then
    echo "=== Creating virtual environment ==="
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
fi
.venv/bin/pip install -q -r requirements.txt

# -----------------------------------------------------------
# Start Streamlit in the background
# -----------------------------------------------------------
echo ""
echo "=== Starting Streamlit on port 8501 — $(date) ==="
.venv/bin/streamlit run app.py \
    --server.headless=true \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    > /cta/users/sait.kacmaz/logs/scidiscover-ui-streamlit-${SLURM_JOB_ID}.log 2>&1 &
STREAMLIT_PID=$!

# -----------------------------------------------------------
# Wait until Streamlit is ready (poll port 8501)
# -----------------------------------------------------------
echo "=== Waiting for Streamlit to be ready ==="
for i in $(seq 1 30); do
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo "=== Streamlit ready after $((i * 2))s ==="
        break
    fi
    sleep 2
done

# -----------------------------------------------------------
# Use cloudflared for the UI tunnel — avoids the ngrok single-static-domain
# limitation (the vLLM job already holds the one free ngrok domain).
# cloudflared quick tunnels are free, require no account, and give a
# unique random *.trycloudflare.com URL every time.
# -----------------------------------------------------------
CLOUDFLARED="$HOME/cloudflared"
if [ ! -f "$CLOUDFLARED" ]; then
    echo "=== Downloading cloudflared ==="
    curl -Lo "$CLOUDFLARED" \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x "$CLOUDFLARED"
fi

CLOUDFLARED_LOG="/cta/users/sait.kacmaz/logs/scidiscover-ui-cloudflared-${SLURM_JOB_ID}.log"

echo ""
echo "=== Starting cloudflare tunnel — $(date) ==="
"$CLOUDFLARED" tunnel --url http://localhost:8501 \
    > "$CLOUDFLARED_LOG" 2>&1 &
CLOUDFLARED_PID=$!

# cloudflared prints the public URL to its log; poll until it appears (up to 60s)
echo "=== Waiting for cloudflare tunnel to establish ==="
TUNNEL_URL=""
for i in $(seq 1 30); do
    TUNNEL_URL=$(grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' \
        "$CLOUDFLARED_LOG" 2>/dev/null | head -1)
    if [ -n "$TUNNEL_URL" ]; then
        echo "=== Tunnel ready after $((i * 2))s ==="
        break
    fi
    sleep 2
done

echo ""
echo "============================================="
echo "=== Streamlit UI public URL: $TUNNEL_URL ==="
echo "============================================="
echo ""

# -----------------------------------------------------------
# Keep the job alive until Streamlit exits
# -----------------------------------------------------------
wait $STREAMLIT_PID
echo "=== Streamlit exited — $(date) ==="
kill $CLOUDFLARED_PID 2>/dev/null
echo "=== UI job finished at $(date) ==="
