export PYTHONPATH="${PYTHONPATH}:$(pwd)"

uv run inspect eval evals/humaneval/humaneval.py --model vllm/google/gemma-3-4b-it --model-base-url http://localhost:8000/v1 --epochs 10 --max-connections 50

# uv run inspect eval evals/humaneval/humaneval.py --model vllm/RedHatAI/gemma-3-4b-it-quantized.w8a8 --model-base-url http://localhost:8000/v1 --epochs 10 --max-connections 50