export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# uv run inspect eval evals/aime2025/aime2025.py --model vllm/Qwen/Qwen3-4B-FP8 --model-base-url http://localhost:8000/v1

uv run inspect eval evals/aime2024/aime2024.py --model vllm/Qwen/Qwen3-4B-FP8 --model-base-url http://localhost:8000/v1