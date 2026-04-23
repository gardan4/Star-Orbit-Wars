"""Path B bot: Gumbel top-k + Sequential Halving over heuristic rollouts.

Integration of `orbitwars.mcts.gumbel_search` behind the `Agent` contract.
On each turn we:
  1. Enumerate per-planet candidate moves via the heuristic's scorer.
  2. Sample K joint actions via the Gumbel top-k trick.
  3. Allocate a rollout budget with Sequential Halving.
  4. Return the highest-mean-Q joint's wire format.

Safety:
  * We stage a heuristic action by EARLY_FALLBACK_MS so a search blow-up
    never results in a no-op turn.
  * Any exception inside search falls back to the staged heuristic move.
  * Rollouts respect an internal hard deadline well below actTimeout.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from orbitwars.bots.base import (
    Action,
    Agent,
    Deadline,
    HARD_DEADLINE_MS,
    SEARCH_DEADLINE_MS,
    no_op,
    obs_get,
)
from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS, HeuristicAgent
from orbitwars.mcts.actions import ActionConfig
from orbitwars.mcts.gumbel_search import GumbelConfig, GumbelRootSearch
from orbitwars.opponent.bayes import ArchetypePosterior


class MCTSAgent(Agent):
    """Gumbel Sequential Halving with heuristic-priored rollouts.

    The agent keeps a single `HeuristicAgent` around as the safe
    fallback. Searches are stateless per call (the GumbelRootSearch
    owns only its RNG).

    Opponent modeling (Path D):
      If ``use_opponent_model`` is True (default), the agent observes
      the opponent's actions each turn and maintains an online
      ArchetypePosterior. The posterior is exposed as
      ``self.opp_posterior`` for diagnostics. A follow-up change will
      route the posterior into MCTS rollouts so search biases toward
      moves that exploit the most-likely archetype — v1 just collects
      the evidence so the data is there when we light up the integration.
    """

    name = "mcts"

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        action_cfg: Optional[ActionConfig] = None,
        gumbel_cfg: Optional[GumbelConfig] = None,
        rng_seed: Optional[int] = None,
        use_opponent_model: bool = True,
    ):
        self.weights = dict(HEURISTIC_WEIGHTS) if weights is None else dict(weights)
        # BOKR-style angle refinement is available (set
        # ``angle_refinement_n_grid > 1`` in your ActionConfig) but
        # DEFAULTED OFF. Smoke testing showed refinement pushes the turn-time
        # tail past Kaggle's 1-second actTimeout (seed=42, default
        # deadline 300ms: max=1156ms, 2 turns over 900ms — forfeit
        # risk). The BOKR module is wired into generate_per_planet_moves
        # so callers can opt in for specific experiments, but the
        # shipped MCTSAgent uses the single-angle behavior to preserve
        # the v3 tail profile (max 882 ms, 0 over 900 ms).
        self.action_cfg = action_cfg or ActionConfig()
        self.gumbel_cfg = gumbel_cfg or GumbelConfig()
        # Arm the decoupled sim-move branch by default. The branch is a
        # no-op unless MCTSAgent also populates ``opp_candidate_builder``
        # with >=2 wires (see _maybe_route_posterior_to_search), so this
        # is backward-compat: behavior only changes once the posterior
        # has concentrated enough to propose a multi-archetype mixture.
        self.gumbel_cfg.use_decoupled_sim_move = True
        self._fallback = HeuristicAgent(weights=self.weights)
        self._search = GumbelRootSearch(
            weights=self.weights,
            action_cfg=self.action_cfg,
            gumbel_cfg=self.gumbel_cfg,
            rng_seed=rng_seed,
        )
        self._use_opponent_model = use_opponent_model
        # Posterior is created lazily on turn 0 so per-match state
        # resets come free with the existing turn-0 reset path below.
        self.opp_posterior: Optional[ArchetypePosterior] = None

        # Posterior telemetry — cheap counters so smokes can reason about
        # WHY a run did or didn't see a use-model delta (vs. a null result
        # with no insight into whether the override ever fired). Fields:
        #   turns_observed   — turns the posterior saw an update this match
        #   override_fires   — turns `opp_policy_override` was set to an archetype
        #   override_clears  — turns we explicitly dropped the override (gate failed)
        #   last_top_name    — most recent argmax archetype (for sanity in logs)
        #   last_top_prob    — most recent max of dist() (0.0 if no posterior yet)
        # Reset on turn 0 along with the other per-match state below.
        self.telemetry: Dict[str, Any] = {
            "turns_observed": 0,
            "override_fires": 0,
            "override_clears": 0,
            "builder_fires": 0,
            "builder_clears": 0,
            "last_top_name": None,
            "last_top_prob": 0.0,
        }

    # Posterior → search override tuning. Conservative: require ~15
    # turns of evidence AND a top-archetype probability at least 2.5x
    # the uniform 1/K baseline. Below that, the posterior is noise.
    _POSTERIOR_MIN_TURNS: int = 15
    _POSTERIOR_MIN_TOP_PROB: float = 0.35
    # Decoupled sim-move branch gate. When the *2nd* archetype also
    # has meaningful mass (>= 0.2 ~= ~1.5x uniform), marginalize over
    # both via decoupled UCB. With second-top below this threshold, a
    # single-archetype SH is strictly stronger (no rollouts wasted on
    # a phantom branch), so we keep the builder = None.
    _POSTERIOR_DECOUPLED_MIN_SECOND_PROB: float = 0.20

    def _maybe_route_posterior_to_search(self) -> None:
        """If the posterior has concentrated, set the search's opponent
        rollout policy to the matching archetype. Otherwise clear any
        prior override."""
        post = self.opp_posterior
        if post is None:
            return
        # Always refresh telemetry when posterior exists, even below the
        # turns gate — telemetry answers "did the smoke run long enough?"
        # which only makes sense if we see turns_observed climb.
        self.telemetry["turns_observed"] = post.turns_observed()
        if post.turns_observed() < self._POSTERIOR_MIN_TURNS:
            return
        dist = post.distribution()
        top_prob = float(dist.max())
        self.telemetry["last_top_prob"] = top_prob
        self.telemetry["last_top_name"] = post.most_likely()
        if top_prob < self._POSTERIOR_MIN_TOP_PROB:
            # Not concentrated → no override (opp rolls under default heuristic).
            if self._search.opp_policy_override is not None:
                self._search.opp_policy_override = None
                self.telemetry["override_clears"] += 1
            # Also make sure the decoupled builder is cleared so the
            # search branch doesn't fire under noise.
            if self._search.opp_candidate_builder is not None:
                self._search.opp_candidate_builder = None
            return
        top_name = post.most_likely()
        # Late-bind the name so every call produces a fresh archetype
        # (HeuristicAgent has per-match state that rollouts must not share).
        # When rollout_policy=="fast", swap in the flavor-matched fast
        # rollout agent — ~30x cheaper per ply, same stylistic bias.
        if self.gumbel_cfg.rollout_policy == "fast":
            from orbitwars.opponent.archetypes import make_fast_archetype
            self._search.opp_policy_override = (
                lambda n=top_name: make_fast_archetype(n)
            )
        else:
            from orbitwars.opponent.archetypes import make_archetype
            self._search.opp_policy_override = (
                lambda n=top_name: make_archetype(n)
            )
        self.telemetry["override_fires"] += 1

        # Decoupled UCB branch: fires only when the *second* archetype
        # also has real mass. Marginalizing over a phantom 2nd branch
        # wastes rollouts, so below the threshold we leave the builder
        # = None and the search falls back to plain Sequential Halving.
        sorted_probs = sorted(dist, reverse=True)
        if (
            len(sorted_probs) >= 2
            and sorted_probs[1] >= self._POSTERIOR_DECOUPLED_MIN_SECOND_PROB
        ):
            self._search.opp_candidate_builder = self._build_opp_candidates
            self.telemetry["builder_fires"] = (
                self.telemetry.get("builder_fires", 0) + 1
            )
        else:
            if self._search.opp_candidate_builder is not None:
                self._search.opp_candidate_builder = None
                self.telemetry["builder_clears"] = (
                    self.telemetry.get("builder_clears", 0) + 1
                )

    def _build_opp_candidates(self, obs: Any, opp_player: int):
        """Compute opp's wire action under each of the top-K archetypes.

        Called by ``GumbelRootSearch`` when the decoupled sim-move branch
        is armed. Returns a list of wire actions — one per archetype —
        that the bandit marginalizes over.

        Fails closed: any exception returns ``[]``, which makes the
        search fall back to plain Sequential Halving (the pre-decoupled
        shipped behavior). This is the contract the search relies on.
        """
        try:
            post = self.opp_posterior
            if post is None:
                return []
            k = max(1, int(self.gumbel_cfg.num_opp_candidates))
            dist = post.distribution()
            # Rank archetypes by posterior mass, descending. Keep only
            # those with non-negligible mass (>= second-prob threshold
            # / 2) so a near-uniform posterior doesn't pad the list
            # with noise candidates.
            floor = 0.5 * self._POSTERIOR_DECOUPLED_MIN_SECOND_PROB
            ranked = sorted(
                [(i, float(p)) for i, p in enumerate(dist)],
                key=lambda ip: -ip[1],
            )
            names = [post.names[i] for i, p in ranked[:k] if p >= floor]
            if len(names) < 2:
                return []

            # Build opp's observation once via a temporary FastEngine
            # (perspective-swap). Cheap — a dict shim + a FastEngine
            # construction, comparable to what search already does
            # per-rollout.
            from orbitwars.engine.fast_engine import FastEngine
            from orbitwars.mcts.gumbel_search import _obs_to_namespace
            from orbitwars.opponent.archetypes import make_archetype

            eng = FastEngine.from_official_obs(
                _obs_to_namespace(obs), num_agents=2,
            )
            opp_obs = eng.observation(opp_player)

            wires = []
            # Fresh Deadline per archetype — generous, since this is
            # called from inside the outer turn budget and the archetype
            # .act()s are cheap heuristic passes (<5 ms each).
            for name in names:
                dl = Deadline()
                try:
                    agent = make_archetype(name)
                    wire = agent.act(opp_obs, dl)
                except Exception:
                    continue
                if isinstance(wire, list):
                    wires.append(wire)
            return wires
        except Exception:
            return []

    def act(self, obs: Any, deadline: Deadline) -> Action:
        # Always stage no_op first so any premature return is legal.
        deadline.stage(no_op())

        # ── Match-start detection MUST precede self._fallback.act() ──
        # Seat 0: obs.step==0 signals a new game.
        # Seat 1: obs.step is None (Kaggle engine quirk); we use
        # next_fleet_id regression (or first-call) as the match-start
        # signal.
        #
        # Detecting BEFORE calling fallback.act is load-bearing: the
        # reset below replaces self._fallback with a fresh HeuristicAgent.
        # If we called self._fallback.act first and then replaced it, the
        # first call's _turn_counter increment (0→1) would be discarded
        # by the replacement, leaving the new fallback's counter at None.
        # On turn 2 its counter then advances None→1 instead of 1→2, so
        # for the remainder of the match fallback._turn_counter is
        # ALWAYS one turn behind a freshly-created HeuristicAgent reading
        # the same observations. MCTS threads that stale counter to
        # search as step_override, so both the anchor heuristic_move AND
        # the search's synthetic obs.step drift off-by-one — which
        # silently breaks anchor-lock at seat 1 (confirmed 3/30 turns
        # diverge by tools/diag_mcts_vs_heur_actions_seat1.py). Seat 0
        # is unaffected because obs.step is authoritative there and
        # HeuristicAgent ignores _turn_counter when raw_step is set.
        raw_step = obs_get(obs, "step", None)
        curr_nfid = int(obs_get(obs, "next_fleet_id", 0))
        if raw_step is not None:
            fresh_game = (int(raw_step) == 0)
        else:
            prev_nfid = getattr(self, "_prev_next_fleet_id", None)
            fresh_game = prev_nfid is None or prev_nfid > curr_nfid
        self._prev_next_fleet_id = curr_nfid
        if fresh_game:
            # Fresh heuristic both for fallback and for the search's
            # internal rollouts.
            self._fallback = HeuristicAgent(weights=self.weights)
            self._search = GumbelRootSearch(
                weights=self.weights,
                action_cfg=self.action_cfg,
                gumbel_cfg=self.gumbel_cfg,
                rng_seed=None,  # fresh RNG; deterministic only if seeded at ctor.
            )
            # Per-match opponent posterior — archetypes are stateful
            # (HeuristicAgent holds _LaunchState), so we reset between games.
            if self._use_opponent_model:
                self.opp_posterior = ArchetypePosterior()
            # Also clear any stale override from the previous match — the
            # new opponent is an unknown, back to default heuristic rollouts.
            self._search.opp_policy_override = None
            self._search.opp_candidate_builder = None
            # Reset per-match telemetry so smokes running back-to-back
            # matches don't see stale counts leaking across games.
            self.telemetry = {
                "turns_observed": 0,
                "override_fires": 0,
                "override_clears": 0,
                "builder_fires": 0,
                "builder_clears": 0,
                "last_top_name": None,
                "last_top_prob": 0.0,
            }

        # Stage the heuristic action as our floor. If search wins, we
        # overwrite; if it doesn't, we return this. The fallback here is
        # guaranteed to be the one we'll keep for this match (fresh-game
        # replacement already happened above), so its _turn_counter
        # stays in lockstep with an outside shadow HeuristicAgent.
        try:
            heuristic_move = self._fallback.act(obs, deadline)
            deadline.stage(heuristic_move)
        except Exception:
            heuristic_move = no_op()

        my_player = int(obs_get(obs, "player", 0))

        # Opponent-model observation. Cheap (<20 ms on a dense mid-game
        # obs) and wrapped in try/except so a defect in the posterior
        # never escapes to the search path. v1 is 2-player only: opp is
        # the other seat.
        #
        # Exploitation: once the posterior has concentrated (>=15 turns
        # observed AND top archetype probability > 0.35, i.e. ~2.5x the
        # uniform 1/7 floor), we route the top archetype's HeuristicAgent
        # as the opponent's rollout policy instead of the generic
        # HeuristicAgent(self.weights). This makes MCTS search under the
        # *actual* inferred opponent model rather than "assume default
        # heuristic". Threshold and grace period are conservative — a
        # wrong override is worse than no override, since search then
        # optimizes against a phantom opponent.
        if self._use_opponent_model and self.opp_posterior is not None:
            try:
                opp_player = 1 - my_player  # 2-player assumption
                self.opp_posterior.observe(obs, opp_player=opp_player)
                self._maybe_route_posterior_to_search()
            except Exception:
                # Posterior is informational-only in v1; a bad update
                # must never break the turn.
                pass

        # Respect the outer agent-level deadline too: if we've already
        # burned most of actTimeout staging the fallback, skip search.
        remaining = deadline.remaining_ms(HARD_DEADLINE_MS)
        if remaining < 50.0:
            return heuristic_move

        # Tighten the search-internal deadline to whatever the outer
        # Deadline gives us, minus:
        #   * _ROLLOUT_OVERSHOOT_BUDGET_MS (260): after sequential_halving's
        #     hard deadline fires, the in-flight rollout can still run its
        #     turn-0 opp-heuristic call + step before the per-ply check in
        #     _rollout_value short-circuits the rest. On dense mid-game
        #     states that overshoot hits ~200-270 ms. Observed (audit pass
        #     2): max turn 1172 ms vs 900 ms outer ceiling → 272 ms
        #     overshoot. Reserve 260 ms so worst case lands under 900 ms.
        #   * 40 ms: post-search wrap-up (action encoding, staging).
        # Without this reservation, a slow pre-search (heuristic.act on a
        # fleet-heavy state + posterior.observe) burns most of the outer
        # budget and the search's internal 300 ms deadline can push total
        # elapsed past 900 ms. The audit measures EXACTLY this number.
        _ROLLOUT_OVERSHOOT_BUDGET_MS = 260.0
        _WRAPUP_BUDGET_MS = 40.0
        safe_budget = min(
            self.gumbel_cfg.hard_deadline_ms,
            remaining - _ROLLOUT_OVERSHOOT_BUDGET_MS - _WRAPUP_BUDGET_MS,
        )
        if safe_budget <= 10.0:
            return heuristic_move

        # Rebuild a one-shot config with the tightened deadline. All other
        # fields (including anchor_improvement_margin!) must be preserved
        # so the safety floor still protects us under the tight budget.
        tight_cfg = GumbelConfig(
            num_candidates=self.gumbel_cfg.num_candidates,
            total_sims=self.gumbel_cfg.total_sims,
            rollout_depth=self.gumbel_cfg.rollout_depth,
            hard_deadline_ms=safe_budget,
            anchor_improvement_margin=self.gumbel_cfg.anchor_improvement_margin,
        )

        # Compute the caller-side outer hard stop: the latest wall-clock
        # instant at which search must return. We reserve
        # _OUTER_CEILING_MARGIN_MS between this stop and HARD_DEADLINE_MS
        # so that an in-flight rollout short-circuiting "one inner
        # iteration after the deadline fires" still lands under the
        # outer actTimeout.
        #
        # _OUTER_CEILING_MARGIN_MS budget:
        #   ~100 ms  — worst-case single-inner-iteration cost in
        #              HeuristicAgent._plan_moves on a dense late-game
        #              state (comments on that loop cite ~100-300 ms for
        #              the full outer iteration; one inner-iteration
        #              slice is the overshoot from a fired deadline).
        #   ~20  ms  — action encoding + deadline.stage + any
        #              in-wrapper gc.collect the harness includes in
        #              the turn-time measurement.
        #   -------
        #    120 ms  — conservative ceiling; tighten once we have
        #              audit data confirming the real pathological
        #              ply cost is lower than 100 ms.
        _OUTER_CEILING_MARGIN_MS = 120.0
        outer_hard_stop_at = (
            time.perf_counter()
            + max(0.0, remaining - _OUTER_CEILING_MARGIN_MS) / 1000.0
        )

        # Wrap the entire swap+search+restore so ANY failure (including
        # attribute access on a broken search object) degrades to the
        # heuristic. Agents in ladder play must never bubble.
        saved_cfg = None
        try:
            saved_cfg = self._search.gumbel_cfg
            self._search.gumbel_cfg = tight_cfg
            t0 = time.perf_counter()
            # Pass the heuristic's move in as the anchor candidate:
            # search will only overwrite it with something evaluated to
            # be better, so the MCTS agent is guaranteed heuristic-or-
            # better in expectation.
            # Thread step from the fallback's turn counter. self._fallback.act
            # was called above and updated its monotonic _turn_counter;
            # we reuse it so search sees the same step even on seat 1
            # (where obs.step is None).
            step_override = getattr(self._fallback, "_turn_counter", None)
            result = self._search.search(
                obs, my_player, start_time=t0,
                anchor_action=heuristic_move,
                outer_hard_stop_at=outer_hard_stop_at,
                step_override=step_override,
            )
        except Exception:
            return heuristic_move
        finally:
            if saved_cfg is not None:
                try:
                    self._search.gumbel_cfg = saved_cfg
                except Exception:
                    pass

        if result is None:
            return heuristic_move

        action = result.best_joint.to_wire()
        deadline.stage(action)
        return action


def build(**overrides) -> MCTSAgent:
    """Factory for packaging / tournament registration."""
    return MCTSAgent(**overrides)
