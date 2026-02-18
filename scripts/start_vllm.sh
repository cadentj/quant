# vllm serve RedHatAI/gemma-3-4b-it-quantized.w8a8 \
    # --dtype bfloat16 \
    # --enable-auto-tool-choice \
    # --tool-call-parser hermes \
    # --gpu-memory-utilization 0.80 \
    # --max_model_len 131072 \

vllm serve google/gemma-3-270m-it \
    --dtype bfloat16 \

# vllm serve unsloth/gemma-3-270m-bnb-4bit