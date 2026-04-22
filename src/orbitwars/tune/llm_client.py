"""LLM clients for EvoTune. Swap-in a real Anthropic / OpenAI adapter.

The protocol is `LLMClient.propose(top_k, n) -> List[str]` — implementations
read the elders' source + scores and return `n` candidate function sources.

We ship two adapters:
  * `MockLLMClient`    — in `evotune.py`, used for tests.
  * `AnthropicClient`  — real call to Claude via the `anthropic` SDK.

Fallback behavior: if the SDK isn't installed or an API call fails, the
client raises a clear RuntimeError. Callers can wrap in their own retry /
backoff; EvoTune's main loop doesn't mask failures so we notice real
outages.

Prompt design:
  * One user turn only — simpler to debug than multi-turn.
  * Shows 2-4 elders with scores; asks for `n` distinct NEW candidates.
  * Explicit sandbox constraints (no imports, no I/O).
  * Explicit return format so we can parse with regex-lite logic.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

from orbitwars.tune.evotune import Candidate, LLMClient


PROMPT_TEMPLATE = """You are evolving a Python scoring function for a Kaggle
RTS bot called "Orbit Wars". Your function assigns a priority to
(source_planet, target_planet) attack pairs; the bot launches the
highest-priority attacks each turn. The goal is to WIN matches against a
fixed opponent pool.

## Function signature

```python
def score(f, w) -> float:
    ...
```

* `f` is a `TargetFeatures` object with these float attributes:
  - target_production  (production rate of target; 1-5)
  - target_defender_now (current defender ship count)
  - projected_defender_at_arrival (defenders when our fleet arrives)
  - is_enemy, is_neutral, is_ally, is_comet  (0 or 1)
  - distance  (Euclidean, 0-140)
  - travel_turns  (turns for our fleet to arrive)
  - ships_to_send
  - source_ships, source_production  (our source planet)
  - step  (current turn, 0-499)
* `w` is a dict of hand-tuned weights. Use `w.get('key', default)`. Keys
  include: w_production, w_ships_cost, w_distance_cost, w_travel_cost,
  mult_neutral, mult_enemy, mult_comet, mult_reinforce_ally, agg_early_game,
  expand_bias.

## Constraints

* Only `math` module and these built-ins: abs, min, max, round, int, float,
  bool, len, sum, any, all, range, sorted, reversed, zip, isinstance.
* NO imports. NO I/O. NO side effects. NO dunder attribute access.
* Return a single float. Higher = more attractive target.
* Runtime budget: 1 ms per call.

## Elders from the previous generation

{elders}

## Your task

Propose {n} NEW candidate functions that might BEAT the elders. Be bold
with structural changes — try:
* Division by `(1 + travel_turns)` or `math.exp(-travel_turns * 0.1)`.
* Ratio of production to projected_defender_at_arrival.
* Sigmoid-like saturations: `1.0 / (1.0 + math.exp(-x))`.
* Non-linear combinations (products, ratios, exponentials).
* Early-game vs late-game branching on `f.step`.
* Penalties for `ships_to_send` exceeding 2x `projected_defender_at_arrival`
  (we're wasting ships).

Return ONLY {n} Python function definitions. Separate each with a blank
line. Each must start with `def score(f, w):` and include a short comment
above it explaining the idea. No markdown fences, no prose.
"""


def _format_elders(top_k: List[Candidate]) -> str:
    if not top_k:
        return "(none — this is the first generation)"
    parts = []
    for i, c in enumerate(top_k):
        parts.append(
            f"### Elder #{i + 1}: win_rate = {c.score:.3f}\n"
            f"```python\n{c.source.strip()}\n```"
        )
    return "\n\n".join(parts)


# Matches exactly one line that starts a `def score(f, w):` function.
_DEF_LINE_RE = re.compile(
    r"^[ \t]*def\s+score\s*\(\s*f\s*,\s*w\s*\)\s*(?:->[^:]*)?:\s*$"
)


def parse_functions(text: str, limit: int) -> List[str]:
    """Extract up to `limit` `def score(f, w):` blocks from LLM text.

    Line-oriented walk: each `def score(f, w):` line opens a new block; the
    block absorbs subsequent blank or deeper-indented lines (the function
    body) and stops at the first non-blank line that dedents back to the
    def's own indent (prose, stray comments, the next def, etc). This keeps
    LLM chatter like "Hope this helps!" out of the captured source.
    Markdown fences are stripped up front.
    """
    text = text.replace("```python", "").replace("```", "")
    lines = text.splitlines()

    def_starts = [i for i, line in enumerate(lines) if _DEF_LINE_RE.match(line)]
    if not def_starts:
        return []

    blocks: List[str] = []
    for idx, start_i in enumerate(def_starts):
        end_i = def_starts[idx + 1] if idx + 1 < len(def_starts) else len(lines)
        block_lines = lines[start_i:end_i]

        def_line = block_lines[0]
        def_indent = len(def_line) - len(def_line.lstrip())

        kept = [def_line]
        for line in block_lines[1:]:
            if not line.strip():
                kept.append(line)
                continue
            line_indent = len(line) - len(line.lstrip())
            if line_indent > def_indent:
                kept.append(line)
                continue
            # Non-blank line at <= def indent — function body has ended.
            break

        while kept and kept[-1].strip() == "":
            kept.pop()

        if kept:
            blocks.append("\n".join(kept))
        if len(blocks) >= limit:
            break
    return blocks


class AnthropicClient:
    """LLMClient backed by Anthropic's Claude.

    Models default to `claude-sonnet-4-5` (high quality, fast for code) but
    can be overridden. Temperature held high (1.0) to encourage diverse
    candidates across the generation.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 1.0,
    ):
        try:
            from anthropic import Anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "anthropic SDK not installed. Install with: "
                "pip install anthropic"
            ) from e

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it or pass api_key=..."
            )
        self._Anthropic = Anthropic
        self.client = Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def propose(self, top_k: List[Candidate], n: int) -> List[str]:
        prompt = PROMPT_TEMPLATE.format(n=n, elders=_format_elders(top_k))
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Response is a list of content blocks; join text blocks.
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        blocks = parse_functions(text, limit=n)
        return blocks


# Sanity: _DEF_LINE_RE smoke
assert _DEF_LINE_RE.match("def score(f, w):") is not None
assert _DEF_LINE_RE.match("    def score(f, w):") is not None
assert _DEF_LINE_RE.match("def score(f, w): return 1.0") is None
