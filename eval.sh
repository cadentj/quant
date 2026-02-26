export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# uv run inspect eval inspect_evals/gdm_intercode_ctf \
#     --model vllm/Qwen/Qwen3-4B-Thinking-2507 \
#     --max-connections 20 \
#     --model-base-url http://localhost:8000/v1 \
#     --epochs 3

uv run inspect eval inspect_evals/swe_bench_verified_mini \
    --model vllm/Qwen/Qwen3-4B-Thinking-2507 \
    --max-connections 20 \
    --model-base-url http://localhost:8000/v1 \
    --epochs 3

# uv run inspect eval inspect_evals/mmlu_pro \
#     --model vllm/Qwen/Qwen3-4B-Thinking-2507 \
#     --max-connections 20 \
#     --model-base-url http://localhost:8000/v1 \
#     --epochs 3