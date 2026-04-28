"""Online Bayesian opponent modeling over archetype portfolio (Path D).

Given the archetype catalogue in ``archetypes.py`` we treat opponent
behavior as a *mixture* over archetypes and maintain a running posterior

    P(archetype = k | observed actions up to turn t)

from which we derive two things:

  (a) A posterior-weighted opponent action distribution used by MCTS
      opponent rollouts (instead of "assume heuristic").
  (b) A bias on our own root prior toward actions that *exploit* the
      most-likely archetype (if the posterior concentrates).

Why Bayesian updating and not a classifier?

  * Classifiers need a training set — we have none at submission time.
    The prior/likelihood combo gives us a *principled* online update
    that works from turn 1 with uniform prior.
  * The posterior's *uncertainty* is the information MCTS needs. A
    classifier returns a point estimate; an opponent who genuinely
    mixes strategies shows up as a flat posterior, and MCTS needs
    that signal to avoid mis-exploiting.

Cost budget:

  Per turn, we evaluate K archetypes (7) on the opponent's obs, each
  costing one ``HeuristicAgent.act()``. Heuristic acts are sub-2 ms.
  7 × 2 ms ≈ 14 ms/turn, well inside the ~5 ms target we'd prefer;
  in practice Python overhead dominates and we see ~10-20 ms. Still
  fits under the MCTS search budget.

Implementation choices:

  * **Log-space updates** — K archetypes × 500 turns × product of
    likelihoods will underflow naive float64 very quickly.
  * **Dirichlet-equivalent interpretation**: we maintain an unnormalized
    log-weight vector ``log_alpha`` and exponentiate on query. This is
    equivalent to a Dirichlet posterior on the mixture weights where
    we treat each turn's observation as drawing one category. The
    temperature knob lets us soften per-turn likelihoods (a real
    opponent is noisier than a pure archetype).
  * **Launch-decision-only likelihood** — for v1 we ignore angle and
    ship-count and match only on "did the opponent launch from planet
    X this turn". Angles are continuous (many approximate matches are
    meaningful) and sizes are dependent on the current ship stockpile
    which varies across archetypes; extending the likelihood to those
    dimensions is a clean follow-up but not needed to separate
    rusher-vs-turtler-vs-harasser.
  * **Per-planet Bernoulli** — each owned planet contributes independent
    evidence. An archetype that correctly predicts launch-vs-hold on
    most planets accumulates posterior mass.

Public surface:

  ArchetypePosterior(archetypes, alpha0=1.0, temperature=2.0, eps=0.1)
      .observe(obs, opp_player)     # call after opp's action is visible
      .distribution() -> np.ndarray # posterior over archetypes
      .most_likely() -> str         # name of highest-posterior archetype
      .reset()                      # new match

Integration sketch:

  post = ArchetypePosterior(all_archetypes())
  for turn in game:
      obs = ...
      if turn > 0:                  # need at least one opp action
          post.observe(obs, opp_player)
      dist = post.distribution()
      # pass into MCTS opponent-rollout mixing
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from orbitwars.bots.base import Deadline, obs_get
from orbitwars.opponent.archetypes import (
    ARCHETYPE_NAMES,
    ArchetypeAgent,
    all_archetypes,
)


# ---- Helpers -----------------------------------------------------------

def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _fabricate_opp_obs(obs: Any, opp_player: int) -> Dict[str, Any]:
    """Orbit Wars is fully observable — same state, different player tag.

    We copy only the fields ``parse_obs`` reads, since feeding the
    archetype an obs that's missing keys it expects would raise.
    """
    return {
        "player": opp_player,
        "step": obs_get(obs, "step", 0),
        "angular_velocity": obs_get(obs, "angular_velocity", 0.0),
        "planets": list(obs_get(obs, "planets", [])),
        "initial_planets": list(obs_get(obs, "initial_planets", [])),
        "fleets": list(obs_get(obs, "fleets", [])),
        "next_fleet_id": obs_get(obs, "next_fleet_id", 0),
        "comet_planet_ids": list(obs_get(obs, "comet_planet_ids", [])),
    }


# ---- Posterior ---------------------------------------------------------

@dataclass
class ArchetypePosterior:
    """Online posterior over archetypes given observed opponent actions.

    Args:
        archetypes: the frozen bots whose log-likelihoods we evaluate.
        alpha0: uniform Dirichlet-prior concentration. Use >1 for a
            stronger "no archetype yet" prior.
        temperature: divides the per-turn log-likelihood before
            accumulation. T=1 is raw Bayes; T>1 softens (noisier
            opponent); T<1 sharpens. We default T=2.0 — real
            opponents rarely match an archetype perfectly.
        eps: per-planet Bernoulli noise floor. An archetype that
            predicts "no launch" but sees launch still contributes
            log(eps) rather than -inf.
    """
    archetypes: List[ArchetypeAgent] = field(default_factory=all_archetypes)
    alpha0: float = 1.0
    temperature: float = 2.0
    eps: float = 0.1
    # Early-exit: once the top archetype's posterior probability reaches
    # this threshold, stop running the K-archetype act() likelihood loop
    # on subsequent turns. Saves ~15 ms/turn (the dominant per-turn cost)
    # once the opponent has been identified. Set to 1.0 to disable.
    # Fleet-id bookkeeping still runs (needed if someone resets us later
    # with a fresh match), and ``turns_observed`` still increments so
    # downstream gates keep working.
    freeze_threshold: float = 0.99

    def __post_init__(self) -> None:
        self.K = len(self.archetypes)
        self.names = [a.name for a in self.archetypes]
        # Log-unnormalized posterior starts at log(alpha0).
        self.log_alpha = np.full(self.K, np.log(self.alpha0), dtype=np.float64)
        # Track previously-seen fleet ids so we can identify new launches.
        self._prev_fleet_ids: Set[int] = set()
        self._last_obs: Optional[Dict[str, Any]] = None
        self._turns_observed: int = 0
        # Frozen once the posterior concentrates past freeze_threshold.
        # While frozen, observe() skips the expensive K-archetype loop.
        self._frozen: bool = False

    # ---- Public ----

    def reset(self) -> None:
        self.log_alpha[:] = np.log(self.alpha0)
        self._prev_fleet_ids.clear()
        self._last_obs = None
        self._turns_observed = 0
        self._frozen = False

    def is_frozen(self) -> bool:
        """True once the posterior concentration crossed ``freeze_threshold``.

        Exposed for smokes/telemetry — lets a test verify the early-exit
        path fired after N turns of strong evidence.
        """
        return self._frozen

    def observe(self, obs: Any, opp_player: int) -> None:
        """Incorporate the opponent's action revealed by ``obs``.

        Must be called in turn order (step increases by 1 each call).
        On the very first call we only snapshot the state; we need the
        previous turn's obs to identify *newly-launched* fleets.
        """
        if self._last_obs is None:
            self._last_obs = obs
            self._prev_fleet_ids = {
                int(f[0]) for f in obs_get(obs, "fleets", [])
            }
            return

        # Early-exit: frozen posterior skips the K-archetype likelihood
        # loop (the ~15 ms/turn hot spot). We keep the fleet-id snapshot
        # current and tick turns_observed so downstream consumers don't
        # see stale telemetry. log_alpha is left untouched — distribution()
        # continues to return the frozen posterior.
        if self._frozen:
            self._prev_fleet_ids = {
                int(f[0]) for f in obs_get(obs, "fleets", [])
            }
            self._last_obs = obs
            self._turns_observed += 1
            return

        # Run the likelihood update path. Tick turns_observed and check
        # for freeze transition regardless of whether the update
        # short-circuits (opp eliminated etc.) — a pre-seeded log_alpha
        # that's already over-threshold should freeze on its first real
        # observe() call.
        self._update_log_alpha(obs, opp_player)
        self._turns_observed += 1
        self._maybe_freeze()

    def _update_log_alpha(self, obs: Any, opp_player: int) -> None:
        """Incorporate one turn of opp evidence into ``log_alpha``.

        Split out from ``observe`` so the freeze check fires at a single
        well-defined point regardless of which control-flow path the
        update took.
        """
        # Identify fleets launched by opp this turn.
        opp_launches = self._opp_launches_this_turn(obs, opp_player)

        # Snapshot current fleet ids for the next turn's diff.
        self._prev_fleet_ids = {
            int(f[0]) for f in obs_get(obs, "fleets", [])
        }

        # Evidence is over *opp-owned planets that exist* on the
        # previous turn's obs — launches come from there. We evaluate
        # each archetype on the previous turn's state (what opp "saw"
        # when deciding), not the current state (which reflects their
        # action + our action + world updates).
        prev_obs = self._last_obs
        self._last_obs = obs

        opp_planet_ids = {
            int(pl[0]) for pl in obs_get(prev_obs, "planets", [])
            if int(pl[1]) == opp_player
        }
        if not opp_planet_ids:
            # Nothing to condition on — opp has been eliminated.
            return

        for k, arch in enumerate(self.archetypes):
            predicted = self._predicted_launches(arch, prev_obs, opp_player)
            log_lik = self._log_likelihood(
                observed_launches=opp_launches,
                predicted_launches=predicted,
                planet_ids=opp_planet_ids,
            )
            self.log_alpha[k] += log_lik / self.temperature

    def _maybe_freeze(self) -> None:
        """Flip ``_frozen`` on when concentration crosses the threshold.

        Called at the end of observe() (non-bootstrap, non-frozen path).
        ``freeze_threshold=1.0`` opts out — the check becomes unreachable.
        """
        if self.freeze_threshold < 1.0:
            if float(_softmax_np(self.log_alpha).max()) >= self.freeze_threshold:
                self._frozen = True

    def distribution(self) -> np.ndarray:
        """Posterior over archetypes as a probability vector."""
        return _softmax_np(self.log_alpha)

    def most_likely(self) -> str:
        return self.names[int(np.argmax(self.log_alpha))]

    def turns_observed(self) -> int:
        return self._turns_observed

    # ---- Internals ----

    def _opp_launches_this_turn(
        self, obs: Any, opp_player: int,
    ) -> Set[int]:
        """Set of planet ids the opponent launched from this turn.

        Uses fleet-id diffing against the previous turn's snapshot. A
        fleet is "new" if its id wasn't in the prior obs.
        """
        launches: Set[int] = set()
        for f in obs_get(obs, "fleets", []):
            fid = int(f[0])
            if fid in self._prev_fleet_ids:
                continue
            owner = int(f[1])
            from_pid = int(f[5])
            if owner == opp_player:
                launches.add(from_pid)
        return launches

    def _predicted_launches(
        self, archetype: ArchetypeAgent, obs: Any, opp_player: int,
    ) -> Set[int]:
        """What set of planets would `archetype` launch from, playing
        for `opp_player` on this obs?"""
        opp_obs = _fabricate_opp_obs(obs, opp_player)
        dl = Deadline()
        action = archetype.act(opp_obs, dl)
        launches: Set[int] = set()
        for mv in action or []:
            if len(mv) >= 1:
                launches.add(int(mv[0]))
        return launches

    def _log_likelihood(
        self,
        observed_launches: Set[int],
        predicted_launches: Set[int],
        planet_ids: Set[int],
    ) -> float:
        """Per-planet Bernoulli log-likelihood.

        For each planet the opponent owned:
          If archetype predicts launch and obs shows launch  → log(1-eps)
          If archetype predicts launch and obs shows hold    → log(eps)
          If archetype predicts hold and obs shows hold      → log(1-eps)
          If archetype predicts hold and obs shows launch    → log(eps)

        We only evaluate on planets the opp actually owns (planet_ids) —
        planets they lost this turn don't carry an action decision.
        """
        if not planet_ids:
            return 0.0
        log_hit = np.log(1.0 - self.eps)
        log_miss = np.log(self.eps)
        total = 0.0
        for pid in planet_ids:
            obs_launch = pid in observed_launches
            pred_launch = pid in predicted_launches
            total += log_hit if (obs_launch == pred_launch) else log_miss
        return total
