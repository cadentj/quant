# %% [markdown]
# # Mock PTQ Figures
#
# This notebook-style script contains simple mock figures for the PTQ story:
# - a grouped bar plot showing that existing PTQ methods look acceptable on
#   basic benchmarks but degrade more on real-world benchmarks
# - a markdown cell with the GPTQ / QuaRot equations and a direction-aware
#   GPTQ loss term
# - two grouped recovery bar plots (probe vs SAE directions) with mock error bars
#
# Benchmark split:
# - Basic benchmarks: `MMLU`, `GSM8K`, `HumanEval`
# - Real-world benchmarks: `SWE-bench Lite`, `AIME 2025`

# %%
import numpy as np
import matplotlib.pyplot as plt


BENCHMARKS = ["MMLU", "GSM8K", "HumanEval", "SWE-bench Lite", "AIME 2025"]
BAR_WIDTH = 0.36
# Indices for real-world benchmarks (used for recovery annotations).
RW_ANNOT_INDICES = (3, 4)

CURRENT_PTQ = {
    "GPTQ": np.array([1.5, -5.0, -4.0, -14.0, -12.5]),
    "QuaRot": np.array([-1.5, -4.0, -4.5, -11.0, -10.5]),
}

# Mock SEs (pp) for illustration only.
CURRENT_PTQ_ERR = {
    "GPTQ": np.array([0.9, 1.0, 0.8, 1.3, 1.2]),
    "QuaRot": np.array([0.8, 0.9, 0.9, 1.1, 1.1]),
}

# Probe-trained direction: slightly stronger recovery on code-heavy tasks.
IMPROVED_PROBE = {
    "GPTQ + probe": np.array([1.2, -4.9, -3.9, -6.2, -5.4]),
    "QuaRot + probe": np.array([-1.4, -4.3, -3.8, -4.6, -4.1]),
}
IMPROVED_PROBE_ERR = {
    "GPTQ + probe": np.array([0.8, 0.9, 0.8, 1.0, 0.95]),
    "QuaRot + probe": np.array([0.7, 0.8, 0.8, 0.9, 0.85]),
}

# SAE-derived direction: similar trend, modestly different real-world recovery.
IMPROVED_SAE = {
    "GPTQ + SAE": np.array([1.0, -5.1, -4.0, -7.4, -6.3]),
    "QuaRot + SAE": np.array([-1.6, -4.6, -4.0, -5.3, -4.8]),
}
IMPROVED_SAE_ERR = {
    "GPTQ + SAE": np.array([0.85, 0.95, 0.85, 1.05, 1.0]),
    "QuaRot + SAE": np.array([0.75, 0.85, 0.85, 0.95, 0.9]),
}


def _pp_diff_vs_baseline(baseline_y: float, improved_y: float) -> str:
    """Signed point change from baseline PTQ to improved (same units as y-axis)."""
    return f"{improved_y - baseline_y:+.1f} pp"


def plot_grouped_bars(
    benchmarks: list[str],
    series: dict[str, np.ndarray],
    title: str,
    errors: dict[str, np.ndarray] | None = None,
) -> None:
    x = np.arange(len(benchmarks))
    fig, ax = plt.subplots(figsize=(10, 5))

    offsets = np.linspace(
        -BAR_WIDTH * (len(series) - 1) / 2,
        BAR_WIDTH * (len(series) - 1) / 2,
        len(series),
    )

    for offset, (label, values) in zip(offsets, series.items(), strict=True):
        bar_kw: dict = {"width": BAR_WIDTH, "label": label}
        if errors is not None and label in errors:
            bar_kw["yerr"] = errors[label]
            bar_kw["capsize"] = 3
            bar_kw["error_kw"] = {
                "elinewidth": 1.0,
                "capthick": 1.0,
                "alpha": 0.85,
            }
        ax.bar(x + offset, values, **bar_kw)

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=15, ha="right")
    ax.set_ylabel("Relative performance change (pp)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(-20, 20)

    labels = list(series.keys())
    if len(labels) == 2:
        a, b = labels[0], labels[1]
        for bi in RW_ANNOT_INDICES:
            va, vb = float(series[a][bi]), float(series[b][bi])
            if va == vb:
                continue
            dpp = va - vb
            xc = 0.5 * ((x[bi] + offsets[0]) + (x[bi] + offsets[1]))
            yc = max(va, vb, 0.0) + 1.2
            ax.annotate(
                f"{dpp:+.1f} pp\n({a} − {b})",
                xy=(xc, yc),
                ha="center",
                va="bottom",
                fontsize=8,
                color="white",
            )

    fig.tight_layout()
    plt.show()


def plot_recovery_bars(
    benchmarks: list[str],
    baseline: dict[str, np.ndarray],
    improved: dict[str, np.ndarray],
    title: str,
    *,
    method_pairs: list[tuple[str, str]],
    colors: tuple[str, str] = ("C0", "C1"),
    baseline_errors: dict[str, np.ndarray] | None = None,
    improved_errors: dict[str, np.ndarray] | None = None,
    improved_hatch: str | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> None:
    x = np.arange(len(benchmarks))
    fig, ax = plt.subplots(figsize=figsize)

    offsets = np.linspace(
        -BAR_WIDTH * (len(method_pairs) - 1) / 2,
        BAR_WIDTH * (len(method_pairs) - 1) / 2,
        len(method_pairs),
    )

    for offset, (baseline_label, improved_label), color in zip(
        offsets, method_pairs, colors, strict=True
    ):
        baseline_values = baseline[baseline_label]
        improved_values = improved[improved_label]

        baseline_bar_kw: dict = {
            "width": BAR_WIDTH,
            "facecolor": "none",
            "edgecolor": color,
            "linewidth": 1.8,
            "linestyle": "--",
            "label": f"{baseline_label} baseline",
        }
        if baseline_errors is not None:
            baseline_bar_kw["yerr"] = baseline_errors[baseline_label]
            baseline_bar_kw["capsize"] = 3
            baseline_bar_kw["error_kw"] = {
                "elinewidth": 1.0,
                "capthick": 1.0,
                "ecolor": color,
                "alpha": 0.85,
            }

        baseline_bar = ax.bar(x + offset, baseline_values, **baseline_bar_kw)

        improved_bar_kw: dict = {
            "width": BAR_WIDTH,
            "color": color,
            "label": improved_label,
        }
        if improved_hatch:
            improved_bar_kw["hatch"] = improved_hatch
            improved_bar_kw["edgecolor"] = "0.35"
            improved_bar_kw["linewidth"] = 0.6
        if improved_errors is not None:
            improved_bar_kw["yerr"] = improved_errors[improved_label]
            improved_bar_kw["capsize"] = 3
            improved_bar_kw["error_kw"] = {
                "elinewidth": 1.0,
                "capthick": 1.0,
                "ecolor": "0.25",
                "alpha": 0.85,
            }

        ax.bar(x + offset, improved_values, **improved_bar_kw)

        # Keep the baseline outline visible on top of the filled bar edge.
        for patch in baseline_bar:
            patch.set_zorder(3)

        for bi in RW_ANNOT_INDICES:
            b_y = float(baseline_values[bi])
            i_y = float(improved_values[bi])
            lbl = _pp_diff_vs_baseline(b_y, i_y)
            bx = x[bi] + offset
            err = 0.0
            if improved_errors is not None:
                err = float(improved_errors[improved_label][bi])
            tip = i_y + err
            ax.annotate(
                lbl,
                xy=(bx, tip),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="white",
            )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=15, ha="right")
    ax.set_ylabel("Relative performance change (pp)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(-20, 20)
    fig.tight_layout()
    plt.show()


# %%
plot_grouped_bars(
    BENCHMARKS,
    CURRENT_PTQ,
    title="W4A4 PTQ: small drops on basic benchmarks, larger drops on real-world tasks",
    errors=CURRENT_PTQ_ERR,
)


# %% [markdown]
# ## GPTQ and QuaRot Equations
#
# Let $W$ be the original weight matrix, $\hat{W}$ the quantized weight matrix,
# and $X$ the calibration activations for a given layer.
#
# Standard GPTQ reconstruction objective:
#
# $$
# \hat{W}_{\mathrm{GPTQ}}
# = \arg\min_{\hat{W}} \left\| X (W - \hat{W}) \right\|_F^2
# $$
#
# Direction-aware GPTQ objective for a single direction $d$:
#
# $$
# \hat{W}_{\mathrm{dir}}
# = \arg\min_{\hat{W}}
# \left\| X (W - \hat{W}) \right\|_F^2
# + \lambda \left\| \bigl(X (W - \hat{W})\bigr) d \right\|_2^2
# $$
#
# Here, $d$ is a direction in the residual stream chosen to capture an
# important failure-relevant behavior. In this setup, $d$ can come from:
# - a learned probe direction for hallucination / incorrect code
# - an SAE-derived direction or feature direction associated with failures
#
# If multiple directions are used, collect them in a matrix $D$ and write:
#
# $$
# \hat{W}_{\mathrm{multi-dir}}
# = \arg\min_{\hat{W}}
# \left\| X (W - \hat{W}) \right\|_F^2
# + \lambda \left\| X (W - \hat{W}) D \right\|_F^2
# $$
#
# QuaRot can be viewed as rotating activations / weights into a more
# quantization-friendly basis, applying quantization in that rotated basis, and
# then using a second GPTQ-style pass for further error reduction. In the
# modified recipe, that second-pass GPTQ objective is replaced with the same
# direction-aware loss above so that quantization also preserves error along
# probe or SAE directions.

# %%
plot_recovery_bars(
    BENCHMARKS,
    CURRENT_PTQ,
    IMPROVED_PROBE,
    title="W4A4 PTQ + learned probe direction: recovery vs baseline PTQ",
    method_pairs=[("GPTQ", "GPTQ + probe"), ("QuaRot", "QuaRot + probe")],
    colors=("C0", "C1"),
    baseline_errors=CURRENT_PTQ_ERR,
    improved_errors=IMPROVED_PROBE_ERR,
)

plot_recovery_bars(
    BENCHMARKS,
    CURRENT_PTQ,
    IMPROVED_SAE,
    title="W4A4 PTQ + SAE feature direction: recovery vs baseline PTQ",
    method_pairs=[("GPTQ", "GPTQ + SAE"), ("QuaRot", "QuaRot + SAE")],
    colors=("#6a4c93", "#1982c4"),
    baseline_errors=CURRENT_PTQ_ERR,
    improved_errors=IMPROVED_SAE_ERR,
    improved_hatch="///",
    figsize=(10, 4.6),
)
