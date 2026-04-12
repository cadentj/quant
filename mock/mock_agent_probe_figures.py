# %% [markdown]
# # Mock agent debugging + probe traces
#
# Notebook-style script: a plausible multi-turn SWE-bench–style debugging trace
# (mock, not live data) and a **synthetic** line plot of two probe outputs vs
# whitespace token index along the streamed assistant + tool text.
#
# **Probe training context (illustrative, not real runs here):**
#
# - **Hallucination probe**: trained on LongFact++-style rollouts (e.g. Qwen
#   completions); Gemini 3 used to label whether each statement is a
#   hallucination; binary targets \(0/1\).
# - **Code correctness probe**: trained on fuzzed HumanEval-style tasks; Gemini 3
#   used to corrupt solutions; binary targets for correct vs incorrect code.
#
# The y-axis values below are **mock log-prob–style scores** (smooth baseline +
# noise + hand-placed bumps), not loaded from trained probes.

# %%
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np

SEED = 11


# Multi-turn mock trace: issue → read → wrong patch → failing pytest →
# hallucinated doc reference → corrected patch → pass.
SWEBENCH_DEBUG_TRANSCRIPT: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "Repository: acme-sortutils (Python).\n"
            "CI failure on main: `tests/test_merge.py::test_merge_sorted`.\n"
            "AssertionError: expected [1, 2, 3, 4, 5] but got [1, 3, 2, 4, 5].\n"
            "Please fix merge so stable sort tie-breaking matches tests."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Plan:\n"
            "- Read `src/sortutils/merge.py` and `tests/test_merge.py`.\n"
            "- Reproduce with `pytest tests/test_merge.py::test_merge_sorted -q`.\n"
            "- Patch `merge` if tie-handling is wrong.\n"
            "Calling read_file on merge implementation."
        ),
    },
    {
        "role": "tool",
        "tool_name": "read_file",
        "content": (
            "# src/sortutils/merge.py\n"
            "def merge(left, right):\n"
            "    out = []\n"
            "    i = j = 0\n"
            "    while i < len(left) and j < len(right):\n"
            "        if left[i] <= right[j]:\n"
            "            out.append(left[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            out.append(right[j])\n"
            "            j += 1\n"
            "    out.extend(left[i:])\n"
            "    out.extend(right[j:])\n"
            "    return out\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The loop compares correctly; the failure is likely from extending "
            "remainder slices in the wrong order when one side exhausts first. "
            "I'll patch: extend `left[i:]` before `right[j:]` so earlier list wins "
            "ties in the remainder path (incorrect reasoning — wrong fix)."
        ),
    },
    {
        "role": "tool",
        "tool_name": "bash",
        "content": (
            "$ pytest tests/test_merge.py::test_merge_sorted -q\n"
            "F\n"
            "tests/test_merge.py:41: AssertionError: expected [1, 2, 3, 4, 5] "
            "but got [1, 3, 2, 4, 5]\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Still failing. Per `docs/internal/sorting-golden.md` (platform "
            "standard for merge tie policy), we must mirror `heapq.merge` index "
            "semantics. That doc also references `DeprecationLedger.get_policy"
            "('merge')` for the canonical rule set.\n"
            "I'll ignore the deprecated helper and instead track explicit "
            "(value, source_index) pairs so left wins on equal keys."
        ),
    },
    {
        "role": "tool",
        "tool_name": "bash",
        "content": (
            "$ pytest tests/test_merge.py::test_merge_sorted -q\n"
            ".\n"
            "1 passed in 0.12s\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Fix: represent items as (value, side, idx) tuples and compare "
            "lexicographically so left precedes right on ties. Tests pass."
        ),
    },
]


def split_word_tokens(text: str) -> list[str]:
    """Non-empty whitespace-separated tokens (word-level token index)."""
    return [t for t in re.split(r"\s+", text.strip()) if t]


def stream_assistant_and_tool_text(turns: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for t in turns:
        role = t["role"]
        if role == "assistant":
            parts.append(t["content"])
        elif role == "tool":
            header = f"[{t.get('tool_name', 'tool')}] "
            parts.append(header + t["content"])
    return "\n\n".join(parts)


def token_spans_for_keywords(
    tokens: list[str],
    keywords: list[str],
    window: int = 2,
) -> list[tuple[int, int]]:
    """Inclusive token index ranges [start, end] to bump/smooth around hits."""
    ranges: list[tuple[int, int]] = []
    lowered = [x.lower() for x in tokens]
    keys = [k.lower() for k in keywords]
    n = len(tokens)
    for i, tok in enumerate(lowered):
        if any(k in tok for k in keys):
            lo = max(0, i - window)
            hi = min(n - 1, i + window)
            ranges.append((lo, hi))
    return ranges


def smooth_bump(
    n: int,
    center: float,
    width: float,
    amplitude: float,
) -> np.ndarray:
    x = np.arange(n, dtype=np.float64)
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def apply_ranges_baseline(
    n: int,
    ranges: list[tuple[int, int]],
    value: float,
) -> np.ndarray:
    y = np.zeros(n, dtype=np.float64)
    for lo, hi in ranges:
        y[lo : hi + 1] = value
    return y


def mock_probe_traces(
    tokens: list[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[tuple[int, int]]]]:
    n = len(tokens)
    x_idx = np.arange(n, dtype=np.float64)

    # Regions: wrong-fix assistant turn (before second pytest) vs hallucinated refs.
    wrong_fix_kw = ["remainder", "wrong", "incorrect", "extend", "`left[i:]`"]
    halluc_kw = [
        "sorting-golden",
        "deprecationledger",
        "get_policy",
        "internal",
        "canonical",
    ]
    correct_fix_kw = ["(value,", "side", "lexicographically", "tuples", "pass"]

    r_wrong = token_spans_for_keywords(tokens, wrong_fix_kw, window=3)
    r_hall = token_spans_for_keywords(tokens, halluc_kw, window=4)
    r_ok = token_spans_for_keywords(tokens, correct_fix_kw, window=3)

    # Hallucination logprob: higher = more suspicious (mock).
    hall = rng.normal(-2.2, 0.11, n)
    for lo, hi in r_hall:
        center = 0.5 * (lo + hi)
        hall += smooth_bump(n, center, width=4.0, amplitude=0.85)
    hall += 0.25 * apply_ranges_baseline(n, r_hall, 1.0)

    # Code correctness logprob: higher = more consistent with correct code (mock).
    code = rng.normal(-1.8, 0.10, n)
    for lo, hi in r_wrong:
        center = 0.5 * (lo + hi)
        code -= smooth_bump(n, center, width=5.0, amplitude=1.1)
    code -= 0.35 * apply_ranges_baseline(n, r_wrong, 1.0)
    for lo, hi in r_ok:
        center = 0.5 * (lo + hi)
        code += smooth_bump(n, center, width=4.5, amplitude=0.75)

    meta = {
        "wrong_fix_ranges": r_wrong,
        "hallucination_ranges": r_hall,
        "correct_fix_ranges": r_ok,
    }
    return hall, code, meta


def plot_probe_traces(
    tokens: list[str],
    hall: np.ndarray,
    code: np.ndarray,
    meta: dict[str, list[tuple[int, int]]],
    title: str,
) -> None:
    n = len(tokens)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(11, 4.2))

    ax.plot(x, hall, color="C0", linewidth=1.2, label="hallucination probe (log-prob)")
    ax.plot(x, code, color="C1", linewidth=1.2, label="code correctness probe (log-prob)")

    def shade_ranges(ranges: list[tuple[int, int]], color: str, alpha: float) -> None:
        for lo, hi in ranges:
            ax.axvspan(lo - 0.5, hi + 0.5, color=color, alpha=alpha, linewidth=0)

    shade_ranges(meta.get("wrong_fix_ranges", []), "C1", 0.08)
    shade_ranges(meta.get("hallucination_ranges", []), "C0", 0.08)

    ax.set_xlabel("Token index (whitespace-split, assistant + tool stream)")
    ax.set_ylabel("Probe log-prob")
    ax.set_title(title)
    ax.set_xlim(-0.5, n - 0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()


# %%
STREAM_TEXT = stream_assistant_and_tool_text(SWEBENCH_DEBUG_TRANSCRIPT)
TOKENS = split_word_tokens(STREAM_TEXT)
RNG = np.random.default_rng(SEED)
HALL_TRACE, CODE_TRACE, META = mock_probe_traces(TOKENS, RNG)

plot_probe_traces(
    TOKENS,
    HALL_TRACE,
    CODE_TRACE,
    META,
    title="Probe log-probs",
)
plt.show()


# %% [markdown]
# The shaded bands mark token windows around **wrong remainder reasoning** (code
# correctness dip) and **invented internal docs / API** strings (hallucination
# bump). The final assistant message about explicit `(value, side, …)` tuples
# lines up with a modest recovery in the mock correctness trace.
