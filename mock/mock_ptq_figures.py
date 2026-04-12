# %% [markdown]
# # Mock PTQ Figures
#
# This notebook-style script contains simple mock figures for the PTQ story:
# - a grouped bar plot showing that existing PTQ methods look acceptable on
#   basic benchmarks but degrade more on real-world benchmarks
# - a markdown cell with the GPTQ / QuaRot equations and a direction-aware
#   GPTQ loss term
# - a second grouped bar plot showing improved real-world performance from a
#   direction-aware PTQ recipe
#
# Benchmark split:
# - Basic benchmarks: `MMLU`, `GSM8K`, `HumanEval`
# - Real-world benchmarks: `SWE-bench Lite`, `AIME 2025`

# %%
import numpy as np
import matplotlib.pyplot as plt


BENCHMARKS = ["MMLU", "GSM8K", "HumanEval", "SWE-bench Lite", "AIME 2025"]
BAR_WIDTH = 0.36

CURRENT_PTQ = {
    "GPTQ": np.array([-4.5, -5.0, -4.0, -14.0, -12.5]),
    "QuaRot": np.array([-3.5, -4.0, -4.5, -11.0, -10.5]),
}

IMPROVED_PTQ = {
    "GPTQ + direction": np.array([-4.0, -4.5, -4.0, -7.0, -6.0]),
    "QuaRot + direction": np.array([-3.0, -3.5, -4.0, -5.0, -4.5]),
}

# Mock standard errors (percentage points) for illustration only.
CURRENT_PTQ_ERR = {
    "GPTQ": np.array([0.8, 0.9, 0.7, 1.2, 1.1]),
    "QuaRot": np.array([0.7, 0.8, 0.8, 1.0, 1.0]),
}
IMPROVED_PTQ_ERR = {
    "GPTQ + direction": np.array([0.7, 0.8, 0.7, 1.0, 0.9]),
    "QuaRot + direction": np.array([0.6, 0.7, 0.8, 0.8, 0.8]),
}


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
    ax.set_ylim(-16, 2)
    fig.tight_layout()
    plt.show()


def plot_recovery_bars(
    benchmarks: list[str],
    baseline: dict[str, np.ndarray],
    improved: dict[str, np.ndarray],
    title: str,
    baseline_errors: dict[str, np.ndarray] | None = None,
    improved_errors: dict[str, np.ndarray] | None = None,
) -> None:
    x = np.arange(len(benchmarks))
    fig, ax = plt.subplots(figsize=(10, 5))

    methods = [("GPTQ", "GPTQ + direction"), ("QuaRot", "QuaRot + direction")]
    offsets = np.linspace(
        -BAR_WIDTH * (len(methods) - 1) / 2,
        BAR_WIDTH * (len(methods) - 1) / 2,
        len(methods),
    )

    for offset, (baseline_label, improved_label) in zip(
        offsets, methods, strict=True
    ):
        baseline_values = baseline[baseline_label]
        improved_values = improved[improved_label]
        color = "C0" if baseline_label == "GPTQ" else "C1"

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

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=15, ha="right")
    ax.set_ylabel("Relative performance change (pp)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(-16, 2)
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
    IMPROVED_PTQ,
    title="W4A4 PTQ with probe / SAE directions: recovery relative to baseline PTQ",
    baseline_errors=CURRENT_PTQ_ERR,
    improved_errors=IMPROVED_PTQ_ERR,
)
