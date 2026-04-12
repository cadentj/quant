# %%
from html import escape
from pathlib import Path
import random
import re


OUTPUT_PATH = Path(__file__).with_suffix(".html")
SEED = 7

BROKEN_CODE = """
def normalize_scores(scores):
    total = sum(scores)
    normalized = []

    for score in scores
        normalized.append(score / total)

    return normalized
""".strip()

BREAKING_DJANGO = """
# urls.py
from django.contrib import admin
from django.urls import path, re_path
from django.http import HttpResponse

def catch_all(request, *args, **kwargs):
    return HttpResponse("maintenance mode")

urlpatterns = [
    re_path(r"^.*$", catch_all),
    path("admin/", admin.site.urls),
]
""".strip()

HALLUCINATION = """
from payments.audit import AuditLogger
from users.permissions import require_workspace_admin
from core.feature_flags import is_rollout_enabled

def delete_workspace(request, workspace_id):
    require_workspace_admin(request.user, workspace_id)

    if not is_rollout_enabled("safe_workspace_delete"):
        return {"error": "feature disabled"}

    AuditLogger.log_workspace_deletion(request.user.id, workspace_id)
    return WorkspaceService.soft_delete_with_snapshot(workspace_id)
""".strip()

SNIPPETS = [
    {
        "title": "Broken Python Syntax",
        "code": BROKEN_CODE,
        "focus_terms": {
            "for": 0.42,
            "score": 0.36,
            "in": 0.48,
            "scores": 0.62,
            "normalized.append(score": 0.93,
        },
        "near_terms": {
            "normalize",
            "total",
            "normalized",
            "append",
            "return",
        },
        "color": (230, 90, 90),
    },
    {
        "title": "Django Core Break",
        "code": BREAKING_DJANGO,
        "focus_terms": {
            "urlpatterns": 0.34,
            "re_path(r\"^.*$\",": 0.66,
            "catch_all),": 0.78,
            "path(\"admin/\",": 0.86,
            "admin.site.urls": 0.71,
        },
        "near_terms": {
            "django",
            "HttpResponse",
            "catch_all",
            "path",
            "re_path",
            "admin",
        },
        "color": (245, 160, 70),
    },
    {
        "title": "LLM Hallucination",
        "code": HALLUCINATION,
        "focus_terms": {
            "payments.audit": 0.63,
            "AuditLogger": 0.58,
            "require_workspace_admin": 0.77,
            "is_rollout_enabled": 0.73,
            "WorkspaceService.soft_delete_with_snapshot": 0.98,
        },
        "near_terms": {
            "workspace",
            "request",
            "feature",
            "delete",
            "error",
            "permissions",
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
  <title>Mock Activation Sequences</title>
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
    <h1>Mock Activation Sequences</h1>
    <p class="intro">
      Tokens are split on whitespace and assigned synthetic SAE-style activation magnitudes:
      mostly near zero, mildly noisy on nearby context, and sharply higher on the tokens most
      related to the syntax break, framework break, or hallucinated code.
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
