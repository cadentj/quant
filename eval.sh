set -euo pipefail
set -a; source .env; set +a

# uv run modal deploy modal/vllm_server.py

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

VLLM_BASE_URL=https://ndif--gemma-4-e2b-it-serve.modal.run/v1


# NOTE(cadentj): I omit 12, 13, 69 since the model sometimes tries to factor large numbers.
# NOTE(cadentj): I also omit 3, 86, since they seem to take a long time.
# NOTE(cadentj): I also omit 60 since the model occasionally hits the limit.
# NOTE(cadentj): I also omit 55, 65, 66, 70 since it contains an image which occasionally hits the 32768 token limit.
uv run inspect eval inspect_evals/gdm_intercode_ctf \
    --model openai/google/gemma-4-E2B-it \
    --max-connections 64 \
    --model-base-url "$VLLM_BASE_URL" \
    -T sandbox=modal \
    --epochs 3

echo "Done!"
