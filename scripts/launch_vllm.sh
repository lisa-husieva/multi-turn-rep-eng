#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for the target model.
# Requires: pip install vllm
# Usage: bash scripts/launch_vllm.sh [model_id] [port] [gpu_memory_utilization]
#
# Examples:
#   bash scripts/launch_vllm.sh
#   bash scripts/launch_vllm.sh meta-llama/Llama-3.1-8B-Instruct 8000 0.9

set -euo pipefail

# Save CUDA_VISIBLE_DEVICES if already set by caller (e.g. CUDA_VISIBLE_DEVICES=5 bash launch_vllm.sh)
_CUDA_CALLER="${CUDA_VISIBLE_DEVICES:-}"

# Load .env for API keys and other settings — but NOT CUDA_VISIBLE_DEVICES
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a && source "$ENV_FILE" && set +a
else
    echo "Warning: .env not found at $ENV_FILE — using environment defaults"
fi

# Restore caller's CUDA_VISIBLE_DEVICES if it was set; otherwise fall back to .env value or default
if [ -n "$_CUDA_CALLER" ]; then
    export CUDA_VISIBLE_DEVICES="$_CUDA_CALLER"
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
fi

MODEL_ID="${1:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${2:-8000}"
GPU_MEM="${3:-0.9}"

echo "Launching vLLM server..."
echo "  Model: $MODEL_ID"
echo "  Port:  $PORT"
echo "  GPU memory utilization: $GPU_MEM"
echo "  CUDA device: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Target API will be available at: http://localhost:$PORT/v1"
echo "Press Ctrl+C to stop."
echo ""

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --api-key "token-abc123"
