# %%
from html import escape
from pathlib import Path
import random
import re


OUTPUT_PATH = Path(__file__).with_suffix(".html")
SEED = 13

# Mock GSM8K-style problems + wrong chain-of-thought (not from the dataset).
# focus_terms must match substrings inside individual whitespace-split tokens.

WRONG_OPERATION = """
Problem: Liam has 50 stickers. He gives away 12 stickers, then buys 8 more.
How many stickers does Liam have now?

When someone gives stickers away, the count should increase by what they gave,
so I first do 50+12=62. Then he buys 8 more: 62+8=70.

#### 70
""".strip()

ENTITY_RATE_CONFUSION = """
Problem: Each of 4 students gets 5 apples. The basket also had 3 extra apples
for the teacher. How many apples were in the basket before handing them out?

Each student gets 5, so 4*5=20 for the class. The teacher is another person,
so I should count the 3 extras once for students and once for the teacher:
20+3+3=26.

#### 26
""".strip()

DISTRACTION_NUMBER = """
Problem: Mia ran 4 laps of 200 meters each. Her friend timed 3 of the laps, and
Mia's total time for those 3 laps was 9 minutes. How many meters did Mia run
in total?

The story mentions 9 minutes and 200 meters, so I multiply them: 9*200=1800.

#### 1800
""".strip()

SNIPPETS = [
    {
        "title": "Wrong operation (inverse / sign)",
        "code": WRONG_OPERATION,
        "focus_terms": {
            "50+12=62": 0.92,
            "increase": 0.72,
            "62+8=70": 0.78,
            "gives": 0.55,
        },
        "near_terms": {
            "stickers",
            "buys",
            "liam",
            "50",
            "12",
            "8",
            "####",
        },
        "color": (230, 90, 90),
    },
    {
        "title": "Entity / double-counting",
        "code": ENTITY_RATE_CONFUSION,
        "focus_terms": {
            "20+3+3=26": 0.95,
            "once": 0.68,
            "teacher": 0.52,
        },
        "near_terms": {
            "apples",
            "students",
            "basket",
            "4*5=20",
            "4",
            "5",
            "3",
            "####",
        },
        "color": (245, 160, 70),
    },
    {
        "title": "Distraction (irrelevant values)",
        "code": DISTRACTION_NUMBER,
        "focus_terms": {
            "9*200=1800": 0.96,
            "minutes": 0.62,
            "1800": 0.88,
        },
        "near_terms": {
            "meters",
            "laps",
            "mia",
            "200",
            "4",
            "3",
            "####",
        },
        "color": (110, 130, 255),
    },
]


# %%
def split_whitespace_tokens(text: str) -> list[str]:
    return re.split(r"(\s+)", text)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def activation_for_token(
    token: str,
    focus_terms: dict[str, float],
    near_terms: set[str],
    rng: random.Random,
) -> float | None:
    if token.isspace():
        return None

    lowered = token.lower()

    for term, target in sorted(
        focus_terms.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if term.lower() in lowered:
            return clamp(rng.gauss(target, 0.045), 0.18, 1.0)

    if any(term in lowered for term in near_terms):
        return clamp(rng.gauss(0.18, 0.08), 0.02, 0.45)

    return clamp(abs(rng.gauss(0.015, 0.02)), 0.0, 0.09)


def render_code_block(
    code: str,
    focus_terms: dict[str, float],
    near_terms: set[str],
    rgb: tuple[int, int, int],
    rng: random.Random,
) -> str:
    pieces: list[str] = []

    for token in split_whitespace_tokens(code):
        activation = activation_for_token(token, focus_terms, near_terms, rng)
        if activation is None:
            pieces.append(escape(token))
            continue

        bg_alpha = 0.08 + 0.55 * activation
        pieces.append(
            "<span class=\"tok\" "
            f"style=\"background: rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {bg_alpha:.3f});\" "
            f"title=\"activation={activation:.2f}\">{escape(token)}</span>"
        )

    return "".join(pieces)


def build_html(snippets: list[dict[str, object]]) -> str:
    rng = random.Random(SEED)
    sections: list[str] = []

    for snippet in snippets:
        code_html = render_code_block(
            code=snippet["code"],
            focus_terms=snippet["focus_terms"],
            near_terms=snippet["near_terms"],
            rgb=snippet["color"],
            rng=rng,
        )
        sections.append(
            f"""
<section>
  <h2>{escape(snippet["title"])}</h2>
  <pre class="code">{code_html}</pre>
</section>
""".strip()
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GSM8K-style reasoning — mock token activations</title>
  <style>
    body {{ margin: 1rem; font-family: system-ui, sans-serif; color: #111; background: #fff; }}
    main {{ max-width: 90ch; margin: 0 auto; }}
    h1 {{ font-size: 1.25rem; margin: 0 0 0.5rem; font-weight: 600; }}
    .intro {{ margin: 0 0 1rem; color: #444; font-size: 0.9rem; line-height: 1.45; }}
    .hint {{ margin: 0 0 1rem; color: #666; font-size: 0.85rem; }}
    section {{ margin-bottom: 1.75rem; }}
    section h2 {{ font-size: 1rem; margin: 0 0 0.35rem; font-weight: 600; }}
    pre.code {{
      margin: 0;
      padding: 0;
      border: none;
      background: transparent;
      font-family: ui-monospace, monospace;
      font-size: 0.8125rem;
      line-height: 1.35;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .tok {{ display: inline; padding: 0; margin: 0; vertical-align: baseline; }}
  </style>
</head>
<body>
  <main>
    <h1>GSM8K-style problems — mock activations</h1>
    <p class="intro">
      Mock grade-school word problems with incorrect chain-of-thought, in the spirit of GSM8K.
      Tokens are split on whitespace; synthetic activations are higher near tokens that touch
      the main reasoning mistake (wrong operation, double-counting, or using a distractor).
    </p>
    <p class="hint">Stronger highlight = higher synthetic activation. Hover a token for the numeric value.</p>
    {''.join(sections)}
  </main>
</body>
</html>
"""


# %%
def main() -> None:
    OUTPUT_PATH.write_text(build_html(SNIPPETS), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
