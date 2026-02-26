set -euo pipefail
set -a; source .env; set +a

# uv run modal deploy modal/vllm_server.py

CONTAINER_ID="${1:?Usage: $0 <modal-container-id>}"

stop_container() {
    echo "Stopping Modal container $CONTAINER_ID..."
    uv run modal container stop "$CONTAINER_ID"
}
trap stop_container EXIT

echo "Starting Modal container $CONTAINER_ID..."

# uv run inspect eval inspect_evals/gdm_intercode_ctf \
#     --model vllm/Qwen/Qwen3-4B-Thinking-2507-FP8 \
#     --max-connections 64 \
#     --model-base-url "$VLLM_URL" \
#     -T sandbox_type=modal \
#     --epochs 1

# uv run inspect eval inspect_evals/swe_bench_verified_mini \
#     --model vllm/Qwen/Qwen3-4B-Thinking-2507-FP8 \
#     --max-connections 64 \
#     --model-base-url "$VLLM_URL" \
#     -T sandbox_type=modal \
#     -T revision=b316c349947c29963fce3f4a65967c9807a4b673 \
#     --epochs 1

uv run inspect eval inspect_evals/tau2_airline \
    --model vllm/Qwen/Qwen3-4B-Thinking-2507-FP8 \
    --max-connections 64 \
    --model-base-url "$VLLM_URL" \
    --epochs 3

echo "Done!"