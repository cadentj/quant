# %%
"""Compare base vs quantized HumanEval results.

Finds task IDs where the base model passes often (>= 0.6) but the
quantized model fails often (<= 0.4), aggregated across epochs.
"""

from collections import defaultdict

from inspect_ai.log import read_eval_log


BASE_LOG = "/root/quant/logs/2026-02-09T22-42-30+00-00_humaneval_WCfsuUNC7CGnQzgfdLQwbT.eval"
QUANT_LOG = "/root/quant/logs/2026-02-09T22-33-58+00-00_humaneval_9HZ4NqRui7jP9VYzeGnQLs.eval"

BASE_THRESHOLD = 0.6   # task must pass at least this often on base
QUANT_THRESHOLD = 0.4  # task must pass at most this often on quantized


def get_pass_rates(log_path: str) -> dict[str | int, float]:
    """Return a dict of task_id -> pass rate (fraction correct across epochs)."""
    log = read_eval_log(log_path)
    correct: dict[str | int, int] = defaultdict(int)
    total: dict[str | int, int] = defaultdict(int)
    if log.samples:
        for sample in log.samples:
            if sample.scores:
                score = next(iter(sample.scores.values()))
                total[sample.id] += 1
                if score.value == "C":
                    correct[sample.id] += 1
    return {tid: correct[tid] / total[tid] for tid in total}


def main() -> None:
    base_rates = get_pass_rates(BASE_LOG)
    quant_rates = get_pass_rates(QUANT_LOG)

    all_ids = sorted(set(base_rates) | set(quant_rates), key=str)

    # Overall stats
    base_n_epochs = sum(1 for _ in read_eval_log(BASE_LOG).samples or [])
    quant_n_epochs = sum(1 for _ in read_eval_log(QUANT_LOG).samples or [])
    print(f"Base:      {len(base_rates)} tasks, {base_n_epochs} total samples")
    print(f"Quantized: {len(quant_rates)} tasks, {quant_n_epochs} total samples")
    print()

    base_avg = sum(base_rates.values()) / len(base_rates) if base_rates else 0
    quant_avg = sum(quant_rates.values()) / len(quant_rates) if quant_rates else 0
    print(f"Base avg pass rate:      {base_avg:.3f}")
    print(f"Quantized avg pass rate: {quant_avg:.3f}")
    print()

    # Find regressions: base >= 0.6 AND quantized <= 0.4
    regressions = []
    for tid in all_ids:
        bp = base_rates.get(tid, 0.0)
        qp = quant_rates.get(tid, 0.0)
        if bp >= BASE_THRESHOLD and qp <= QUANT_THRESHOLD:
            regressions.append((tid, bp, qp))

    regressions.sort(key=lambda r: (r[2] - r[1], str(r[0])))  # worst regressions first

    print(
        f"Regressions (base >= {BASE_THRESHOLD}, quantized <= {QUANT_THRESHOLD}): "
        f"{len(regressions)}"
    )
    print(f"{'Task ID':<25} {'Base':>8} {'Quant':>8} {'Delta':>8}")
    print("-" * 51)
    for tid, bp, qp in regressions:
        print(f"  {tid:<23} {bp:>7.1%} {qp:>7.1%} {qp - bp:>+7.1%}")


if __name__ == "__main__":
    main()