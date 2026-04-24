"""Decoupled bandits for simultaneous-move root nodes.

Orbit Wars is a **simultaneous-move** game — each turn both players
submit their actions without seeing the other's. Plain UCT / PUCT
assumes a max-player / min-player tree (sequential), which is
*unsound* for sim-move games: the opponent's response is not yet
committed when we pick ours, so we shouldn't credit / discredit a
candidate my-move based on a single assumed opp-move.

The principled fix at the root is a pair of **decoupled bandits**:
  * My player runs UCB (or Exp3) over my candidate actions.
  * The opponent runs UCB (or Exp3) over their candidate actions.
  * Both players pick simultaneously each sim; a single rollout
    evaluates the joint (my, opp) cell and returns v ∈ [-1, +1]
    from my player's perspective.
  * Stats update: my[i] += v, opp[j] += -v. Each side maximizes
    their own marginal mean Q.

The key property: my candidate's estimated Q is computed by
*averaging over all opp responses tried so far*, weighted by how
often opp picked them — i.e. the empirical opp strategy. This is
the correct way to score my-move under uncertainty about opp's
response, as opposed to the baseline single-opp-heuristic rollout
that assumes the opp plays one fixed deterministic action.

Why it matters here:
  * The baseline `sequential_halving` in gumbel_search.py uses a
    fixed heuristic opp rollout policy. If the real opp deviates
    from heuristic — and with the Bayesian posterior's archetype
    portfolio we expect them to — our Q estimates are biased.
  * Decoupled UCB at the root automatically marginalizes over
    opp-uncertainty: more opp candidates = more thorough integration
    over opp's strategy.

Scope (v1):
  * Root-only. Non-root tree expansion arrives alongside BOKR and a
    neural prior (W4-5). For now we trust the heuristic rollout from
    the leaf joint cell.
  * UCB1 bandit per player. Exp3 / regret-matching variant sketched
    but not wired (flag-gated in future work).
  * Simple budget allocation: total_sims rollouts allocated greedily
    by decoupled UCB. No Sequential Halving at this layer — it's
    incompatible with the 2D bandit structure, and the UCB exploration
    bonus gives us good simple-regret bounds already.

Integration:
  * GumbelRootSearch.search() can call ``decoupled_ucb_root()`` in
    place of ``sequential_halving`` when the ``use_decoupled_sim_move``
    flag is set on GumbelConfig. The caller is responsible for
    supplying ``opp_candidates`` (typically sampled from the posterior-
    biased heuristic or from 2-3 top archetypes).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from orbitwars.mcts.actions import JointAction


# ----------------------------- Result type -------------------------------

@dataclass
class DecoupledSearchResult:
    """Outcome of a decoupled-bandit root search.

    best_my_joint — the my-candidate with highest marginal mean Q.
    n_rollouts    — total (my, opp) cells evaluated.
    duration_ms   — wall time spent inside the bandit loop.
    my_q_values   — per-my-candidate marginal mean Q (my-perspective).
    my_visits     — per-my-candidate visit count.
    opp_q_values  — per-opp-candidate marginal mean Q (opp-perspective,
                    i.e. averaged -v_my).
    opp_visits    — per-opp-candidate visit count.
    aborted       — True if the wall-clock deadline cut the loop short.
    """
    best_my_joint: JointAction
    n_rollouts: int
    duration_ms: float
    my_q_values: List[float] = field(default_factory=list)
    my_visits: List[int] = field(default_factory=list)
    opp_q_values: List[float] = field(default_factory=list)
    opp_visits: List[int] = field(default_factory=list)
    aborted: bool = False

    @property
    def n_my_candidates(self) -> int:
        return len(self.my_q_values)

    @property
    def n_opp_candidates(self) -> int:
        return len(self.opp_q_values)


# ----------------------------- UCB bandit --------------------------------

def _ucb_select(
    q_sum: List[float],
    visits: List[int],
    total: int,
    c: float,
) -> int:
    """UCB1 selection: argmax_i (mean_q + c * sqrt(log(N)/n_i)).

    Unvisited arms (visits[i]==0) get infinity, so every arm is pulled
    at least once before we start exploiting. Ties broken by index.
    """
    best_i = 0
    best_score = -math.inf
    log_n = math.log(max(1, total))
    for i in range(len(q_sum)):
        n = visits[i]
        if n == 0:
            # Unvisited — highest priority. Return first unvisited.
            return i
        mean = q_sum[i] / n
        bonus = c * math.sqrt(log_n / n)
        score = mean + bonus
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def decoupled_ucb_root(
    my_candidates: List[Any],
    opp_candidates: List[Any],
    rollout_fn: Callable[[Any, Any], float],
    total_sims: int,
    hard_deadline_ms: float,
    c_my: float = 1.4,
    c_opp: float = 1.4,
    start_time: Optional[float] = None,
    protected_my_idx: Optional[int] = None,
) -> DecoupledSearchResult:
    """Decoupled UCB1 over (my, opp) candidates at the root.

    Each iteration:
      i = my-UCB arg-max     (my-player exploitation + exploration)
      j = opp-UCB arg-max    (opp-player exploitation + exploration)
      v = rollout_fn(my[i], opp[j])   # my-perspective, ∈ [-1, +1]
      my_q[i]  += v
      opp_q[j] += -v
      visits[i], visits[j] += 1

    Returns the my-candidate with highest marginal mean-Q after the
    budget is exhausted.

    protected_my_idx (if given) is guaranteed at least one rollout at
    every opp candidate before standard UCB selection kicks in — same
    role as sequential_halving's ``protected_idx``, used for the anchor
    (heuristic fallback) candidate. Ensures a low-variance Q estimate
    for the anchor so the anchor_guard in mcts_bot can do its job.

    c_my / c_opp are UCB exploration coefficients. 1.4 ≈ sqrt(2) is
    the textbook default for bounded-reward UCB1; we let them diverge
    in case tuning finds asymmetric values (e.g. smaller c_opp if the
    opp-candidate pool is already well-targeted via posterior).

    Deadline semantics match sequential_halving: we check
    time.perf_counter() at the top of every iteration and abort
    (setting ``aborted=True``) if we're past the hard deadline. A
    single in-flight rollout can still overshoot by its own cost —
    the caller is responsible for budgeting that via rollout_fn
    (e.g. the same deadline_fn wiring _rollout_value uses).
    """
    t0 = start_time if start_time is not None else time.perf_counter()
    deadline = t0 + hard_deadline_ms / 1000.0

    n_my = len(my_candidates)
    n_opp = len(opp_candidates)
    if n_my == 0:
        raise ValueError("decoupled_ucb_root: no my candidates")
    if n_opp == 0:
        raise ValueError("decoupled_ucb_root: no opp candidates")

    my_q_sum: List[float] = [0.0] * n_my
    my_visits: List[int] = [0] * n_my
    opp_q_sum: List[float] = [0.0] * n_opp
    opp_visits: List[int] = [0] * n_opp

    total_rollouts = 0
    aborted = False
    # Iterate over the *joint* cross product until each cell has at
    # least one visit, then switch to UCB. This is equivalent to the
    # "play each arm once" warm-up of classic UCB1 but extended to 2D.
    #
    # protected_my_idx skips to playing the anchor against every opp
    # first — gives the anchor's marginal Q the lowest variance before
    # we start exploiting. Matches sequential_halving's ``protected_idx``.
    warmup_pairs: List[Tuple[int, int]] = []
    if protected_my_idx is not None and 0 <= protected_my_idx < n_my:
        for j in range(n_opp):
            warmup_pairs.append((protected_my_idx, j))
    for i in range(n_my):
        if i == protected_my_idx:
            continue
        for j in range(n_opp):
            warmup_pairs.append((i, j))

    # Phase 1: warm-up. One rollout per (i, j) cell.
    for (i, j) in warmup_pairs:
        if total_rollouts >= total_sims:
            break
        if time.perf_counter() > deadline:
            aborted = True
            break
        v = rollout_fn(my_candidates[i], opp_candidates[j])
        my_q_sum[i] += v
        my_visits[i] += 1
        opp_q_sum[j] += -v
        opp_visits[j] += 1
        total_rollouts += 1

    # Phase 2: UCB exploitation — each player picks independently.
    while total_rollouts < total_sims and not aborted:
        if time.perf_counter() > deadline:
            aborted = True
            break
        i = _ucb_select(my_q_sum, my_visits, total_rollouts, c_my)
        j = _ucb_select(opp_q_sum, opp_visits, total_rollouts, c_opp)
        v = rollout_fn(my_candidates[i], opp_candidates[j])
        my_q_sum[i] += v
        my_visits[i] += 1
        opp_q_sum[j] += -v
        opp_visits[j] += 1
        total_rollouts += 1

    def _mean(i: int, sums: List[float], visits: List[int]) -> float:
        return sums[i] / visits[i] if visits[i] > 0 else -math.inf

    # Best my candidate by marginal mean Q — this is the action we play.
    best_my_i = max(range(n_my), key=lambda i: _mean(i, my_q_sum, my_visits))

    my_q_avg = [_mean(i, my_q_sum, my_visits) for i in range(n_my)]
    opp_q_avg = [_mean(j, opp_q_sum, opp_visits) for j in range(n_opp)]

    return DecoupledSearchResult(
        best_my_joint=my_candidates[best_my_i],
        n_rollouts=total_rollouts,
        duration_ms=(time.perf_counter() - t0) * 1000.0,
        my_q_values=my_q_avg,
        my_visits=list(my_visits),
        opp_q_values=opp_q_avg,
        opp_visits=list(opp_visits),
        aborted=aborted,
    )


# ----------------------------- Exp3 variant ------------------------------
# Regret-matching / Exp3 is the principled fallback when UCB's mean
# estimates are too noisy under sim-move. Kept as a sketch; wire when
# we have enough empirical evidence that UCB is underperforming.

def decoupled_exp3_root(
    my_candidates: List[JointAction],
    opp_candidates: List[JointAction],
    rollout_fn: Callable[[JointAction, JointAction], float],
    total_sims: int,
    hard_deadline_ms: float,
    eta: float = 0.3,
    start_time: Optional[float] = None,
    protected_my_idx: Optional[int] = None,
    rng: Optional["_r.Random"] = None,  # noqa: F821  (lazy type ref)
) -> DecoupledSearchResult:
    """Decoupled Exp3 — softmax-weighted picks with importance-weighted updates.

    Per iteration:
      p_my[i]  ∝ exp(eta * cum_my[i])       (softmax over log-gains)
      p_opp[j] ∝ exp(eta * cum_opp[j])
      i ~ p_my,  j ~ p_opp
      v = rollout_fn(my[i], opp[j])
      cum_my[i]  += v / p_my[i]             # importance weighting
      cum_opp[j] += (-v) / p_opp[j]

    Exp3 is minimax-optimal for adversarial bandits and is the canonical
    regret-matching variant for sim-move games. Trades UCB's simplicity
    for robustness when the opp is adversarial or non-stationary — both
    plausible on the Kaggle ladder.

    eta is the learning rate. 0.3 is a safe default for [-1, +1] rewards
    and budgets in the 16-128 range; tune later.

    protected_my_idx (if given) is guaranteed at least one rollout at
    every opp candidate before the softmax draw kicks in — same role as
    decoupled_ucb_root's ``protected_my_idx`` / sequential_halving's
    ``protected_idx``. This is load-bearing for the anchor_guard in
    mcts_bot: without it, a low-total_sims EXP3 run may never visit the
    heuristic anchor and return a garbage marginal Q that the guard
    can't distinguish from "actually worse than anchor".

    rng (optional): pre-seeded ``random.Random`` instance. If ``None``
    we allocate a fresh unseeded instance — fine for production where
    we want per-search variance, but tests should pass a seeded RNG
    for determinism.
    """
    import random as _r
    if rng is None:
        rng = _r.Random()

    t0 = start_time if start_time is not None else time.perf_counter()
    deadline = t0 + hard_deadline_ms / 1000.0

    n_my = len(my_candidates)
    n_opp = len(opp_candidates)
    if n_my == 0 or n_opp == 0:
        raise ValueError("decoupled_exp3_root: empty candidate list")

    cum_my = [0.0] * n_my
    cum_opp = [0.0] * n_opp
    my_q_sum = [0.0] * n_my
    my_visits = [0] * n_my
    opp_q_sum = [0.0] * n_opp
    opp_visits = [0] * n_opp

    total_rollouts = 0
    aborted = False

    def _softmax(cum: List[float]) -> List[float]:
        m = max(cum)
        es = [math.exp(eta * (c - m)) for c in cum]
        s = sum(es)
        return [e / s for e in es]

    def _sample(p: List[float]) -> int:
        r = rng.random()
        cumulative = 0.0
        for i, pi in enumerate(p):
            cumulative += pi
            if r <= cumulative:
                return i
        return len(p) - 1

    # Phase 0: anchor warm-up. Mirror decoupled_ucb_root's protected
    # warm-up so the anchor's marginal Q is always well-estimated before
    # the softmax has a chance to "forget" it. We still importance-weight
    # under p_my/p_opp even during warm-up, keeping the Exp3 regret bound
    # intact — the draw is just forced.
    if protected_my_idx is not None and 0 <= protected_my_idx < n_my:
        for j in range(n_opp):
            if total_rollouts >= total_sims:
                break
            if time.perf_counter() > deadline:
                aborted = True
                break
            p_my = _softmax(cum_my)
            p_opp = _softmax(cum_opp)
            v = rollout_fn(my_candidates[protected_my_idx], opp_candidates[j])
            cum_my[protected_my_idx] += v / max(p_my[protected_my_idx], 1e-6)
            cum_opp[j] += (-v) / max(p_opp[j], 1e-6)
            my_q_sum[protected_my_idx] += v
            my_visits[protected_my_idx] += 1
            opp_q_sum[j] += -v
            opp_visits[j] += 1
            total_rollouts += 1

    while total_rollouts < total_sims and not aborted:
        if time.perf_counter() > deadline:
            aborted = True
            break
        p_my = _softmax(cum_my)
        p_opp = _softmax(cum_opp)
        i = _sample(p_my)
        j = _sample(p_opp)
        v = rollout_fn(my_candidates[i], opp_candidates[j])
        # Importance-weighted cumulative gains.
        cum_my[i] += v / max(p_my[i], 1e-6)
        cum_opp[j] += (-v) / max(p_opp[j], 1e-6)
        my_q_sum[i] += v
        my_visits[i] += 1
        opp_q_sum[j] += -v
        opp_visits[j] += 1
        total_rollouts += 1

    # Best my by mean-Q (not the softmax distribution — at play time we
    # commit to a single action, not a mixed strategy).
    def _mean(i: int, sums: List[float], visits: List[int]) -> float:
        return sums[i] / visits[i] if visits[i] > 0 else -math.inf

    best_my_i = max(range(n_my), key=lambda i: _mean(i, my_q_sum, my_visits))

    return DecoupledSearchResult(
        best_my_joint=my_candidates[best_my_i],
        n_rollouts=total_rollouts,
        duration_ms=(time.perf_counter() - t0) * 1000.0,
        my_q_values=[_mean(i, my_q_sum, my_visits) for i in range(n_my)],
        my_visits=list(my_visits),
        opp_q_values=[_mean(j, opp_q_sum, opp_visits) for j in range(n_opp)],
        opp_visits=list(opp_visits),
        aborted=aborted,
    )
