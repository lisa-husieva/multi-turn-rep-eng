#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for the target model.
# Requires: pip install vllm
# Usage: bash scripts/launch_vllm.sh [model_id] [port] [gpu_memory_utilization]
#
# Examples:
#   bash scripts/launch_vllm.sh
#   bash scripts/launch_vllm.sh meta-llama/Llama-3.1-8B-Instruct 8000 0.9

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

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
