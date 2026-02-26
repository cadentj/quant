export HF_HOME=/root/quant/.cache
export HF_XET_HIGH_PERFORMANCE=1

uv run vllm serve Qwen/Qwen3-4B-Thinking-2507 \
    --dtype bfloat16 \
    --max-model-len "131072" \