# %% [markdown]
# # Mock Ablation Figures
#
# Side-by-side: probe direction (left) vs SAE direction (right).
# All results are the ablated (worse) variants, degraded ~2-8 pp from vanilla
# PTQ baseline to show what happens without the key component.

# %%
import numpy as np
import matplotlib.pyplot as plt

BENCHMARKS = ["MMLU", "GSM8K", "HumanEval", "SWE-bench Lite", "AIME 2025"]
BAR_WIDTH = 0.36
RW_ANNOT_INDICES = (3, 4)

CURRENT_PTQ = {
    "GPTQ": np.array([1.5, -5.0, -4.0, -14.0, -12.5]),
    "QuaRot": np.array([-1.5, -4.0, -4.5, -11.0, -10.5]),
}
CURRENT_PTQ_ERR = {
    "GPTQ": np.array([0.9, 1.0, 0.8, 1.3, 1.2]),
    "QuaRot": np.array([0.8, 0.9, 0.9, 1.1, 1.1]),
}

# --- Ablated probe direction (fixed λ=1, no per-layer tuning) ---
# 2-8 pp worse than baseline PTQ (CURRENT_PTQ).

ABLATED_PROBE = {
    "GPTQ + probe": np.array([-0.5, -7.5, -6.0, -18.0, -17.5]),
    "QuaRot + probe": np.array([-4.0, -7.0, -7.0, -16.5, -15.0]),
}
ABLATED_PROBE_ERR = {
    "GPTQ + probe": np.array([0.9, 1.0, 0.9, 1.2, 1.1]),
    "QuaRot + probe": np.array([0.8, 0.9, 0.9, 1.1, 1.0]),
}

# --- Ablated SAE direction (random direction instead of learned feature) ---
# 2-8 pp worse than baseline PTQ.

ABLATED_SAE = {
    "GPTQ + SAE": np.array([-1.5, -7.0, -7.0, -17.5, -16.5]),
    "QuaRot + SAE": np.array([-5.0, -6.5, -6.5, -15.0, -15.5]),
}
ABLATED_SAE_ERR = {
    "GPTQ + SAE": np.array([0.95, 1.0, 0.9, 1.15, 1.1]),
    "QuaRot + SAE": np.array([0.85, 0.9, 0.9, 1.05, 1.0]),
}


def _pp_diff_vs_baseline(baseline_y: float, improved_y: float) -> str:
    return f"{improved_y - baseline_y:+.1f} pp"


def _draw_recovery_panel(
    ax: plt.Axes,
    benchmarks: list[str],
    baseline: dict[str, np.ndarray],
    improved: dict[str, np.ndarray],
    title: str,
    *,
    method_pairs: list[tuple[str, str]],
    colors: tuple[str, str],
    baseline_errors: dict[str, np.ndarray] | None = None,
    improved_errors: dict[str, np.ndarray] | None = None,
    improved_hatch: str | None = None,
) -> None:
    x = np.arange(len(benchmarks))

    offsets = np.linspace(
        -BAR_WIDTH * (len(method_pairs) - 1) / 2,
        BAR_WIDTH * (len(method_pairs) - 1) / 2,
        len(method_pairs),
    )

    for offset, (bl, il), color in zip(offsets, method_pairs, colors, strict=True):
        bv = baseline[bl]
        iv = improved[il]

        bkw: dict = {
            "width": BAR_WIDTH,
            "facecolor": "none",
            "edgecolor": color,
            "linewidth": 1.8,
            "linestyle": "--",
            "label": f"{bl} baseline",
        }
        if baseline_errors is not None:
            bkw["yerr"] = baseline_errors[bl]
            bkw["capsize"] = 3
            bkw["error_kw"] = {
                "elinewidth": 1.0,
                "capthick": 1.0,
                "ecolor": color,
                "alpha": 0.85,
            }
        bb = ax.bar(x + offset, bv, **bkw)

        ikw: dict = {"width": BAR_WIDTH, "color": color, "alpha": 0.5, "label": il}
        if improved_hatch:
            ikw["hatch"] = improved_hatch
            ikw["edgecolor"] = "0.35"
            ikw["linewidth"] = 0.6
        if improved_errors is not None:
            ikw["yerr"] = improved_errors[il]
            ikw["capsize"] = 3
            ikw["error_kw"] = {
                "elinewidth": 1.0,
                "capthick": 1.0,
                "ecolor": "0.25",
                "alpha": 0.85,
            }
        ax.bar(x + offset, iv, **ikw)

        for patch in bb:
            patch.set_zorder(3)

        for bi in RW_ANNOT_INDICES:
            b_y = float(bv[bi])
            i_y = float(iv[bi])
            lbl = _pp_diff_vs_baseline(b_y, i_y)
            bx = x[bi] + offset
            err = float(improved_errors[il][bi]) if improved_errors else 0.0
            tip = min(i_y, b_y) - abs(err) - 0.5
            ax.annotate(
                lbl,
                xy=(bx, tip),
                xytext=(0, -5),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7,
                color=color,
            )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Relative performance change (pp)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(-25, 10)


# %% Side-by-side: probe ablation (left) vs SAE ablation (right)
fig, (ax_probe, ax_sae) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

_draw_recovery_panel(
    ax_probe,
    BENCHMARKS,
    CURRENT_PTQ,
    ABLATED_PROBE,
    "Ablation: probe direction (fixed λ = 1)",
    method_pairs=[("GPTQ", "GPTQ + probe"), ("QuaRot", "QuaRot + probe")],
    colors=("C0", "C1"),
    baseline_errors=CURRENT_PTQ_ERR,
    improved_errors=ABLATED_PROBE_ERR,
)

_draw_recovery_panel(
    ax_sae,
    BENCHMARKS,
    CURRENT_PTQ,
    ABLATED_SAE,
    "Ablation: SAE direction (random direction)",
    method_pairs=[("GPTQ", "GPTQ + SAE"), ("QuaRot", "QuaRot + SAE")],
    colors=("#6a4c93", "#1982c4"),
    baseline_errors=CURRENT_PTQ_ERR,
    improved_errors=ABLATED_SAE_ERR,
    improved_hatch="///",
)

ax_sae.set_ylabel("")
fig.suptitle(
    "Ablation: degraded direction-aware PTQ variants",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
plt.show()
