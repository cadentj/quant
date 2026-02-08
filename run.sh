DEPTH=2
SEED=0

uv run parity.py \
    width=256 \
    depth=$DEPTH \
    seed=$SEED \
    dtype=bfloat16 \
    codes="[[0], [1], [2], [3], [0, 1, 2, 3]]" \
    samples_per_task=2000 \
    steps=200000 \
    verbose=True \
    save_dir=/test \
    wandb_project=math-capstone \