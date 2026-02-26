export PYTHONPATH="${PYTHONPATH}:$(pwd)"

uv run inspect eval inspect_evals/gdm_intercode_ctf \
    --model vllm/Qwen/Qwen3-4B-Thinking-2507-FP8 \
    --max-connections 20 \
    --model-base-url "$VLLM_URL" \
    -T sandbox_type=modal \
    --epochs 3

uv run inspect eval inspect_evals/swe_bench_verified_mini \
    --model vllm/Qwen/Qwen3-4B-Thinking-2507-FP8 \
    --max-connections 20 \
    --model-base-url "$VLLM_URL" \
    -T sandbox_type=modal \
    -T revision=b316c349947c29963fce3f4a65967c9807a4b673 \
    --epochs 3

uv run inspect eval inspect_evals/tau2_airline \
    --model vllm/Qwen/Qwen3-4B-Thinking-2507-FP8 \
    --max-connections 20 \
    --model-base-url "$VLLM_URL" \
    --epochs 3
