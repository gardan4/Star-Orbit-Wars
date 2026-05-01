"""Path D exploitation smoke — opponent model vs. no opponent model.

Design:
  For each archetype X in the portfolio, run two match groups:
    A) MCTSAgent(use_opponent_model=False, margin=0.5) vs X
    B) MCTSAgent(use_opponent_model=True,  margin=0.5) vs X
  Both seats, both seeds. Compare win-rate. If the posterior-driven
  opponent model is working, Group B wins more often.

Why margin=0.5 and not the shipped default (2.0)?
  With margin=2.0 the wire action is always the heuristic's — search
  results can't influence behavior, so the opponent model provides no
  end-to-end Elo. The ablation is meaningful only once search actually
  steers the output. margin=0.5 was the W2 best-single-seed default
  and is a plausible regime where a sharper opponent model could help.
  This smoke is explicitly a research experiment, not a leaderboard run.

Caveat:
  Under 1-s CPU budget we're already at the edge; the low-sim regime
  masks small Elo differences in noise. Run N_SEEDS × 2 seats × 2
  groups × N_ARCHETYPES games — total cost roughly
    wall/game × 2 × 2 × 7 ≈ 56 × wall/game
  so with 200-s games this is a ~3-hour experiment. Start small.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import List

from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.mcts.gumbel_search import GumbelConfig
from orbitwars.opponent.archetypes import ARCHETYPE_NAMES, make_archetype
from tournaments.harness import play_game


@dataclass
class Cell:
    archetype: str
    use_model: bool
    wins: int = 0
    losses: int = 0
    ties: int = 0
    # Posterior telemetry accumulated across games in this cell.
    total_override_fires: int = 0
    total_override_clears: int = 0
    total_turns_observed: int = 0
    # Track how many games saw at least one override fire. A cell with
    # games_with_override == 0 is a null cell: the posterior never
    # concentrated enough to matter, so any use-model vs no-model delta
    # is noise, not a real exploitation signal.
    games_with_override: int = 0
    last_top_names: List[str] = field(default_factory=list)

    def record(self, diff: int, telemetry: dict = None) -> None:
        if diff > 0:
            self.wins += 1
        elif diff < 0:
            self.losses += 1
        else:
            self.ties += 1
        if telemetry is not None:
            fires = int(telemetry.get("override_fires", 0))
            self.total_override_fires += fires
            self.total_override_clears += int(telemetry.get("override_clears", 0))
            self.total_turns_observed += int(telemetry.get("turns_observed", 0))
            if fires > 0:
                self.games_with_override += 1
            name = telemetry.get("last_top_name")
            if name:
                self.last_top_names.append(name)

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


def _mk_mcts(
    use_model: bool,
    margin: float,
    total_sims: int = 32,
    num_candidates: int = 4,
    rollout_depth: int = 15,
    hard_deadline_ms: float = 300.0,
    rollout_policy: str = "heuristic",
) -> MCTSAgent:
    return MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=num_candidates,
            total_sims=total_sims,
            rollout_depth=rollout_depth,
            hard_deadline_ms=hard_deadline_ms,
            anchor_improvement_margin=margin,
            rollout_policy=rollout_policy,
        ),
        rng_seed=0,
        use_opponent_model=use_model,
    )


def run_match(
    mcts_seat: int, archetype: str, use_model: bool, margin: float, seed: int,
    total_sims: int = 32, num_candidates: int = 4, rollout_depth: int = 15,
    hard_deadline_ms: float = 300.0, rollout_policy: str = "heuristic",
) -> tuple:
    """Returns (score diff, mcts telemetry dict).

    Telemetry is a snapshot of MCTSAgent.telemetry at end-of-match — the
    per-match reset means this reflects only this game.
    """
    mcts = _mk_mcts(
        use_model, margin,
        total_sims=total_sims,
        num_candidates=num_candidates,
        rollout_depth=rollout_depth,
        hard_deadline_ms=hard_deadline_ms,
        rollout_policy=rollout_policy,
    )
    opp = make_archetype(archetype)
    agents = [mcts.as_kaggle_agent(), opp.as_kaggle_agent()]
    if mcts_seat == 1:
        agents = [opp.as_kaggle_agent(), mcts.as_kaggle_agent()]
    result = play_game(agents, seed=seed, players=2, step_timeout=2.0)
    diff = int(result.final_scores[mcts_seat]) - int(
        result.final_scores[1 - mcts_seat]
    )
    # Snapshot-copy the dict — not the reference — since the MCTSAgent
    # instance is about to go out of scope.
    telem = dict(mcts.telemetry)
    return diff, telem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    ap.add_argument(
        "--archetypes", nargs="+", type=str,
        default=["rusher", "turtler", "defender"],
    )
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument(
        "--total-sims", type=int, default=32,
        help="MCTS total rollout budget per turn",
    )
    ap.add_argument(
        "--num-candidates", type=int, default=4,
        help="Gumbel top-k candidates at the root",
    )
    ap.add_argument(
        "--rollout-depth", type=int, default=15,
        help="Plies simulated per rollout",
    )
    ap.add_argument(
        "--hard-deadline-ms", type=float, default=300.0,
        help="Wall-clock ceiling per MCTS turn",
    )
    ap.add_argument(
        "--rollout-policy", type=str, default="heuristic",
        choices=["heuristic", "fast"],
        help="Which policy fills rollout plies. 'heuristic' is the "
        "shipped mcts_v1 default (full-strength heuristic ~3 ms/call); "
        "'fast' is FastRolloutAgent (~8 us/call, ~380x faster), which "
        "gives ~13x more rollouts at the same deadline. Expected to "
        "unlock the opp-model exploitation signal that was masked by "
        "the 2-sims/turn budget under 'heuristic'.",
    )
    args = ap.parse_args()

    cells: List[Cell] = []
    print(f"Archetypes: {args.archetypes}")
    print(f"Seeds: {args.seeds}")
    print(f"Margin: {args.margin}")
    print(f"Rollout policy: {args.rollout_policy}")
    print(f"Total games: {len(args.archetypes) * len(args.seeds) * 2 * 2}")
    print()

    t_start = time.perf_counter()
    for arch in args.archetypes:
        for use_model in (False, True):
            cell = Cell(archetype=arch, use_model=use_model)
            for seed in args.seeds:
                for seat in (0, 1):
                    diff, telem = run_match(
                        seat, arch, use_model, args.margin, seed,
                        total_sims=args.total_sims,
                        num_candidates=args.num_candidates,
                        rollout_depth=args.rollout_depth,
                        hard_deadline_ms=args.hard_deadline_ms,
                        rollout_policy=args.rollout_policy,
                    )
                    cell.record(diff, telemetry=telem)
                    tag = "+M" if use_model else "  "
                    # Per-game posterior summary — helps debug null results
                    # (did the posterior concentrate? did it pick right?).
                    post_info = ""
                    if use_model:
                        fires = telem.get("override_fires", 0)
                        top = telem.get("last_top_name") or "-"
                        post_info = (
                            f" [post: fires={fires:>3d} top={top:>12s} "
                            f"p={telem.get('last_top_prob', 0.0):.2f}]"
                        )
                    print(
                        f"  arch={arch:>13s} {tag} seed={seed} seat={seat}: "
                        f"diff={diff:+5d}{post_info}",
                        flush=True,
                    )
            cells.append(cell)
            extra = ""
            if use_model and cell.games:
                avg_fires = cell.total_override_fires / cell.games
                extra = (
                    f" | avg_fires={avg_fires:.1f}/game "
                    f"games_with_override={cell.games_with_override}/{cell.games}"
                )
            print(
                f"  -> {arch} use_model={use_model} "
                f"W{cell.wins}/L{cell.losses}/T{cell.ties} "
                f"wr={cell.win_rate:.2f}{extra}",
                flush=True,
            )

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal wall: {elapsed:.0f} s")
    print()
    print("Summary (per archetype):")
    for arch in args.archetypes:
        no_m = next(c for c in cells if c.archetype == arch and not c.use_model)
        wm = next(c for c in cells if c.archetype == arch and c.use_model)
        delta = wm.win_rate - no_m.win_rate
        print(
            f"  {arch:>13s}: no_model wr={no_m.win_rate:.2f}  "
            f"with_model wr={wm.win_rate:.2f}  delta={delta:+.2f}"
        )


if __name__ == "__main__":
    main()
