"""Bridge between EvoTune's abstract scorer protocol and ``HeuristicAgent``.

What EvoTune evolves is the **priority function** — a mapping
``TargetFeatures × weights → float`` that ranks candidate target planets
per owned planet. Everything else (angle/travel-time computation,
defender projection, trajectory obstruction check, exact-plus-one sizing)
stays the well-tested baseline code in ``heuristic.py``. This bridge:

  * Extracts ``TargetFeatures`` from the same local context that
    ``_score_target`` sees so the LLM has a clean, flat feature dict.
  * Compiles the candidate source in the EvoTune sandbox (same
    ``compile_scorer`` function the tests exercise) and wraps it into a
    drop-in replacement for ``heuristic._score_target``.
  * Installs that replacement via module-attribute reassignment. We do
    not subclass ``HeuristicAgent`` because ``_score_target`` is a
    module-level function closed over by the agent's ``act`` method —
    subclassing would require refactoring ``heuristic.py``, which we
    don't want to do for an experiment.

Worker semantics:

  On Windows ``multiprocessing.Pool`` uses ``spawn``, meaning each
  worker re-imports ``orbitwars.bots.heuristic`` from scratch. A
  monkey-patch in the parent is therefore *not* visible to children.
  The fitness path in ``fitness.py`` uses ``Pool(initializer=...)`` to
  run ``install_evo_scorer`` once per worker; that way every game the
  worker plays uses the evolved scorer without per-task setup cost.

Thread-safety: this module writes to ``heuristic._score_target``, which
is a shared module attribute. Do NOT use it in threaded code — spawn
only. The tests exercise install/uninstall serially.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from orbitwars.bots import heuristic as _H
from orbitwars.tune.evotune import (
    TargetFeatures,
    UnsafeCandidateError,
    compile_scorer,
)

# ---- Saved-original slot ------------------------------------------------

# We intentionally keep this as a module global so callers can install and
# uninstall cleanly across generations. Storing the original on the first
# install guards against double-install corrupting the pointer.
_ORIGINAL_SCORE_TARGET: Optional[Callable[..., Tuple[float, float, int]]] = None


def _features_from_context(
    mp: List[Any],
    tp: List[Any],
    ip: List[Any],
    po: "_H.ParsedObs",
    table: "_H.ArrivalTable",
    ships_to_send: int,
    turns: float,
    projected: int,
) -> TargetFeatures:
    """Build a ``TargetFeatures`` from the fields the baseline
    ``_score_target`` already computed.

    All fields are float32-safe — no enum/string leaks into the scorer.
    Booleans are 0/1 floats so ``f.is_enemy * w['alpha']`` expressions
    work without the LLM having to guess a coercion.
    """
    is_comet = 1.0 if tp[0] in po.comet_planet_ids else 0.0
    is_ally = 1.0 if tp[1] == po.player else 0.0
    is_neutral = 1.0 if tp[1] == -1 else 0.0
    # Enemy = not ally and not neutral (and not unowned sun/comet variant);
    # comets may be neutral OR enemy-captured — both are fine, mult is
    # separate.
    is_enemy = 1.0 if (tp[1] != po.player and tp[1] != -1) else 0.0

    distance = math.hypot(tp[2] - mp[2], tp[3] - mp[3])

    return TargetFeatures(
        target_production=float(tp[6]),
        target_defender_now=float(tp[5]),
        projected_defender_at_arrival=float(projected),
        is_enemy=is_enemy,
        is_neutral=is_neutral,
        is_ally=is_ally,
        is_comet=is_comet,
        distance=float(distance),
        travel_turns=float(turns),
        ships_to_send=float(ships_to_send),
        source_ships=float(mp[5]),
        source_production=float(mp[6]),
        step=float(po.step),
    )


def _make_evo_score_target(
    scorer: Callable[[TargetFeatures, Dict[str, float]], float],
) -> Callable[..., Tuple[float, float, int]]:
    """Build a drop-in replacement for ``heuristic._score_target``.

    Keeps the baseline flow for angle, travel time, sun-avoidance, and
    defender projection. Swaps the *score formula* for the evolved
    scorer. Capture-infeasibility penalty is preserved — otherwise a
    naïve evolved scorer that ignores ``ships_to_send`` would get
    credited for sending 1-ship attacks on fortified planets.
    """
    # We recompute travel-turns + intercept-position + sun-route locally
    # (rather than calling a helper) to match the baseline's exact
    # sequencing in case future refactors split the function.

    def _score_target_evo(
        mp: List[Any],
        tp: List[Any],
        ip: List[Any],
        po: "_H.ParsedObs",
        table: "_H.ArrivalTable",
        weights: Dict[str, float],
        ships_to_send: int,
    ) -> Tuple[float, float, int]:
        source_center = (float(mp[2]), float(mp[3]))
        source_radius = float(mp[4])

        angle, turns = _H._travel_turns(
            source_center, tp, ip,
            po.angular_velocity, po.step, ships_to_send,
            source_radius=source_radius, po=po,
        )
        if turns <= 0 or math.isinf(turns):
            return (-math.inf, 0.0, 0)

        target_pos = _H._intercept_position(
            source_center, tp, ip, po.angular_velocity, po.step, turns, po=po,
        )
        angle = _H.route_angle_avoiding_sun(source_center, angle, target_pos)

        defender_ships = tp[5]
        defender_owner = tp[1]
        production = tp[6]
        arrival_turn = po.step + int(math.ceil(turns))
        projected = table.projected_defender_at(
            tp[0], defender_owner, defender_ships, production, arrival_turn,
        )

        feats = _features_from_context(
            mp, tp, ip, po, table, ships_to_send, turns, projected,
        )
        # Compiled scorer already returns -inf on internal exception.
        s = scorer(feats, weights)
        if not math.isfinite(s):
            return (-math.inf, angle, projected)

        # Keep the baseline's "can't capture" penalty. Without it, the
        # evolved scorer can freely recommend under-sized attacks that
        # burn ships; with it, every candidate is evaluated on the same
        # feasibility-aware scale so the LLM learns to *size* rather
        # than to bluff.
        needed = projected + int(weights.get("ships_safety_margin", 1))
        if ships_to_send < needed and defender_owner != po.player:
            s -= 10.0

        return (float(s), float(angle), int(projected))

    return _score_target_evo


# ---- Install / uninstall API -------------------------------------------

def install_evo_scorer(source: str) -> None:
    """Compile ``source`` (must define ``score(features, weights)``) and
    install it as the active ``heuristic._score_target``.

    Idempotent w.r.t. saving the original — repeated installs don't lose
    the baseline pointer. Raises :class:`UnsafeCandidateError` or
    :class:`SyntaxError` on bad source; the original stays installed in
    that case.
    """
    global _ORIGINAL_SCORE_TARGET
    scorer = compile_scorer(source)  # may raise; original untouched
    if _ORIGINAL_SCORE_TARGET is None:
        _ORIGINAL_SCORE_TARGET = _H._score_target
    _H._score_target = _make_evo_score_target(scorer)


def uninstall_evo_scorer() -> None:
    """Restore the baseline ``_score_target``. Safe to call if nothing
    is installed (no-op)."""
    global _ORIGINAL_SCORE_TARGET
    if _ORIGINAL_SCORE_TARGET is not None:
        _H._score_target = _ORIGINAL_SCORE_TARGET
        _ORIGINAL_SCORE_TARGET = None


def is_installed() -> bool:
    """True iff an evolved scorer is currently active. Useful for tests
    that want to verify install/uninstall round-trips."""
    return _ORIGINAL_SCORE_TARGET is not None


# ---- Pool initializer --------------------------------------------------

def _worker_init_evo_scorer(source: str) -> None:
    """Top-level initializer so ``multiprocessing.Pool(initializer=...)``
    can pickle it.

    Called once per worker at pool startup. Compiles the source,
    monkey-patches ``heuristic._score_target`` for the worker's lifetime,
    and leaves it installed — no uninstall needed because the worker
    process exits at the end of the pool's life.
    """
    try:
        install_evo_scorer(source)
    except (UnsafeCandidateError, SyntaxError) as e:
        # In a worker we can't raise usefully — the failure would show
        # up as a cryptic BrokenPipeError on the parent. Instead, leave
        # the baseline scorer in place and let the parent score the
        # candidate at the baseline (which is harmless — it just means
        # the candidate is indistinguishable from baseline, so it won't
        # survive selection).
        print(f"[worker] evo scorer install failed: {e}", flush=True)


__all__ = [
    "install_evo_scorer",
    "uninstall_evo_scorer",
    "is_installed",
    "_worker_init_evo_scorer",
    "_features_from_context",
    "_make_evo_score_target",
]
