set -euo pipefail
set -a; source .env; set +a

# uv run modal deploy modal/vllm_server.py

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

VLLM_BASE_URL=https://ndif--gemma-4-e2b-it-serve.modal.run/v1


# NOTE(cadentj): I omit 12, 13, 69 since the model sometimes tries to factor large numbers.
uv run inspect eval inspect_evals/gdm_intercode_ctf \
    --model openai/google/gemma-4-E2B-it \
    --max-connections 64 \
    --model-base-url "$VLLM_BASE_URL" \
    -T sandbox=modal \
    --epochs 3

echo "Done!"
