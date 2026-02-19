DEPTH=2
SEED=0

uv run parity.py \
    seed=$SEED \
    dtype=float32 \
    samples_per_task=2000 \
    steps=50000 \
    verbose=True \
    save_dir=./results \
    wandb_project=math-capstone \
    model.width=256 \
    model.depth=$DEPTH \
    model.activation=ReLU \
    model.layernorm=false \
    task.n=64 \
    task.n_tasks=4 \
    "task.codes=[[0], [1], [2], [3], [0, 1, 2, 3]]"


    # steps=200000 \