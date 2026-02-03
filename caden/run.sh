DEPTH=2
SEED=0

uv run parity.py \
    --width 128 \
    --depth $DEPTH \
    --seed $SEED \
    --codes "[[0], [1], [2], [3], [0, 1, 2, 3]]" \
    --samples-per-task 2000 \
    --steps 200000 \
    --verbose \
    --save-dir /test