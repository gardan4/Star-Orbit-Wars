"""Tests for EvoTune LLM client infrastructure.

Focus is on the parse layer (LLM output → list of Python sources) — the
actual Anthropic call needs an API key and costs money so it's wrapped in
a skip-if-no-key integration test.
"""
from __future__ import annotations

import os

import pytest

from orbitwars.tune.evotune import Candidate, compile_scorer
from orbitwars.tune.llm_client import (
    PROMPT_TEMPLATE,
    _format_elders,
    parse_functions,
)


def test_parse_functions_extracts_single_block():
    text = (
        "Here's a candidate:\n"
        "def score(f, w):\n"
        "    return f.target_production / (1.0 + f.travel_turns)\n"
    )
    blocks = parse_functions(text, limit=1)
    assert len(blocks) == 1
    assert "def score(f, w):" in blocks[0]
    # Must be compilable.
    compile_scorer(blocks[0])


def test_parse_functions_extracts_multiple_blocks_separated_by_blanks():
    text = (
        "# Candidate A: simple production / travel\n"
        "def score(f, w):\n"
        "    return f.target_production / (1.0 + f.travel_turns)\n"
        "\n"
        "# Candidate B: production squared\n"
        "def score(f, w):\n"
        "    return f.target_production * f.target_production\n"
    )
    blocks = parse_functions(text, limit=3)
    assert len(blocks) == 2
    # Each block contains its own `def score(f, w):`.
    assert all("def score(f, w):" in b for b in blocks)
    # Each block compiles in isolation.
    for b in blocks:
        compile_scorer(b)


def test_parse_functions_strips_markdown_fences():
    """LLMs sometimes wrap code in ```python ... ``` — parser must ignore."""
    text = (
        "```python\n"
        "def score(f, w):\n"
        "    return 1.0\n"
        "```\n"
    )
    blocks = parse_functions(text, limit=1)
    assert len(blocks) == 1
    compile_scorer(blocks[0])


def test_parse_functions_respects_limit():
    text = "\n\n".join(
        f"def score(f, w):\n    return {i}.0\n" for i in range(5)
    )
    blocks = parse_functions(text, limit=3)
    assert len(blocks) == 3


def test_parse_functions_tolerates_noise_around():
    text = (
        "Sure! Here are my proposals.\n\n"
        "def score(f, w):\n"
        "    return f.target_production\n\n"
        "Hope this helps!\n"
    )
    blocks = parse_functions(text, limit=2)
    assert len(blocks) == 1
    compile_scorer(blocks[0])


def test_parse_functions_returns_empty_on_garbage():
    assert parse_functions("nothing to see here", limit=3) == []


def test_format_elders_empty_shows_first_generation_banner():
    assert "first generation" in _format_elders([]).lower()


def test_format_elders_includes_score_and_source():
    c = Candidate(source="def score(f, w): return 1.0", score=0.73)
    s = _format_elders([c])
    assert "0.730" in s
    assert "def score(f, w): return 1.0" in s


def test_prompt_template_has_required_placeholders():
    """Renders without KeyError."""
    rendered = PROMPT_TEMPLATE.format(n=5, elders="(example)")
    assert "5" in rendered
    assert "(example)" in rendered


# ---- Integration (runs only with an API key) ----

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live LLM call"
)
def test_anthropic_client_returns_valid_candidates():
    """Optional: hit the real API, assert it returns parseable functions.
    Gated behind env var so CI / offline dev stays fast + free."""
    from orbitwars.tune.llm_client import AnthropicClient
    client = AnthropicClient()
    candidates = client.propose(top_k=[], n=2)
    assert len(candidates) >= 1
    for c in candidates:
        compile_scorer(c)
