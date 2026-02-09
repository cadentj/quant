vllm serve Qwen/Qwen3-4B-FP8 \
    --dtype bfloat16 \
    # --enable-auto-tool-choice \
    # --tool-call-parser hermes \
    # --gpu-memory-utilization 0.80 \
    # --max_model_len 131072 \