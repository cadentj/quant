set -euo pipefail
set -a; source .env; set +a

# OpenAI client requires a key; vLLM ignores it — set if missing.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# uv run modal deploy modal/vllm_server.py

VLLM_BASE_URL=https://ndif--gemma-4-e2b-it-serve.modal.run/v1

uv run inspect eval-retry logs/2026-04-12T00-26-11+00-00_gdm-intercode-ctf_mynjcEZUUXRamjGrpzdZ52.eval

# Inspect uses the first path segment as the provider: `google/...` = Gemini API (needs google-genai).
# Use `openai/...` + --model-base-url so traffic goes to vLLM; the served name is still google/gemma-4-E2B-it.
# uv run inspect eval-retry inspect_evals/gdm_intercode_ctf \
#     --model openai/google/gemma-4-E2B-it \
#     --max-connections 64 \
#     --model-base-url "$VLLM_BASE_URL" \
#     -T sandbox_type=modal \
#     --epochs 1

# uv run inspect eval inspect_evals/swe_bench_verified_mini \
#     --model vllm/Qwen/Qwen3-4B-Thinking-2507 \
#     --max-connections 64 \
#     --model-base-url "$VLLM_BASE_URL" \
#     -T sandbox_type=modal \
#     -T revision=b316c349947c29963fce3f4a65967c9807a4b673 \
#     --epochs 1

# uv run inspect eval inspect_evals/tau2_airline \
#     --model vllm/Qwen/Qwen3-4B-Thinking-2507-FP8 \
#     --max-connections 64 \
#     --model-base-url "$VLLM_BASE_URL" \
#     --epochs 3

echo "Done!"
