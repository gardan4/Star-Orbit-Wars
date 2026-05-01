"""W4 Exp3 vs UCB head-to-head gate at the simultaneous-move root.

Runs MCTS-exp3 against MCTS-ucb for N games with alternating seats and
varying seeds. Both bots share the same GumbelConfig modulo
``sim_move_variant``; all other knobs (total_sims, num_candidates,
rollout_depth, hard_deadline_ms, use_decoupled_sim_move, etc.) stay
identical so the only treatment effect is the bandit rule at the
decoupled root when the opponent posterior has concentrated.

Gate (plan §W4, "regret-matching A/B test at sim-move nodes — ship if
beats decoupled-UCT by >=5pp"): ship variant="exp3" in the next bundle
iff exp3's win rate >= 0.55 over N games.

Usage:
    python tools/ab_exp3_vs_ucb.py --games 20 --step-timeout 1.0 \\
        --hard-deadline-ms 300 --out runs/ab_exp3_vs_ucb.json

Reporting:
    * raw W/L/D and win-rate for exp3 (ties counted as 0.5)
    * Elo delta (K=16 nominal)
    * turn-time p50/p95/max per variant
    * per-seed table so the caller can sanity-check concentration
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple

from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.mcts.gumbel_search import GumbelConfig
from tournaments.harness import play_game


@dataclass
class GameRow:
    game_idx: int
    seed: int
    exp3_seat: int          # 0 or 1: which seat exp3 played this game
    reward_exp3: int        # {-1, 0, +1} from exp3's perspective
    reward_ucb: int
    steps: int
    wall_s: float
    p50_exp3_ms: float
    p95_exp3_ms: float
    max_exp3_ms: float
    p50_ucb_ms: float
    p95_ucb_ms: float
    max_ucb_ms: float


def _pct(vals: List[float], p: float) -> float:
    if not vals:
        return float("nan")
    v = sorted(vals)
    k = max(0, min(len(v) - 1, int(math.ceil(p / 100 * len(v))) - 1))
    return v[k]


def _make_cfg(variant: str, total_sims: int, num_cand: int,
              rollout_depth: int, hard_deadline_ms: float,
              exp3_eta: float) -> GumbelConfig:
    cfg = GumbelConfig(
        num_candidates=num_cand,
        total_sims=total_sims,
        rollout_depth=rollout_depth,
        hard_deadline_ms=hard_deadline_ms,
    )
    cfg.sim_move_variant = variant
    cfg.exp3_eta = exp3_eta
    # Leave use_decoupled_sim_move at the MCTSAgent default (True) so
    # variant actually takes effect when the opp-model proposes >=2
    # opp candidates. This is the only path where variant matters.
    return cfg


def play_one(seed: int, exp3_seat: int, step_timeout: float,
             total_sims: int, num_cand: int, rollout_depth: int,
             hard_deadline_ms: float, exp3_eta: float,
             verbose: bool = True) -> GameRow:
    """Play one 500-turn game and record exp3-centric metrics."""
    exp3_cfg = _make_cfg("exp3", total_sims, num_cand, rollout_depth,
                         hard_deadline_ms, exp3_eta)
    ucb_cfg = _make_cfg("ucb", total_sims, num_cand, rollout_depth,
                        hard_deadline_ms, exp3_eta)

    # Distinct rng seeds per seat so the two bots are not coupled through
    # their own Gumbel rng.
    exp3_agent = MCTSAgent(gumbel_cfg=exp3_cfg, rng_seed=seed)
    ucb_agent = MCTSAgent(gumbel_cfg=ucb_cfg, rng_seed=seed ^ 0xDEADBEEF)

    if exp3_seat == 0:
        agents = [exp3_agent.as_kaggle_agent(), ucb_agent.as_kaggle_agent()]
    else:
        agents = [ucb_agent.as_kaggle_agent(), exp3_agent.as_kaggle_agent()]

    gc.collect()
    t0 = time.perf_counter()
    result = play_game(agents, seed=seed, players=2, step_timeout=step_timeout)
    wall = time.perf_counter() - t0

    tt0 = result.turn_times_ms[0]
    tt1 = result.turn_times_ms[1]
    if exp3_seat == 0:
        exp3_tt, ucb_tt = tt0, tt1
    else:
        exp3_tt, ucb_tt = tt1, tt0

    row = GameRow(
        game_idx=-1,
        seed=seed,
        exp3_seat=exp3_seat,
        reward_exp3=result.rewards[exp3_seat],
        reward_ucb=result.rewards[1 - exp3_seat],
        steps=result.steps,
        wall_s=wall,
        p50_exp3_ms=statistics.median(exp3_tt) if exp3_tt else float("nan"),
        p95_exp3_ms=_pct(exp3_tt, 95),
        max_exp3_ms=max(exp3_tt) if exp3_tt else float("nan"),
        p50_ucb_ms=statistics.median(ucb_tt) if ucb_tt else float("nan"),
        p95_ucb_ms=_pct(ucb_tt, 95),
        max_ucb_ms=max(ucb_tt) if ucb_tt else float("nan"),
    )
    if verbose:
        out = "W" if row.reward_exp3 == 1 else ("L" if row.reward_exp3 == -1 else "D")
        print(f"  seed={seed:>4}  exp3={exp3_seat}  {out} "
              f"steps={row.steps:>3}  wall={wall:>5.1f}s  "
              f"p50 exp3={row.p50_exp3_ms:>4.0f}ms ucb={row.p50_ucb_ms:>4.0f}ms",
              flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20,
                    help="total games (split across both seats)")
    ap.add_argument("--step-timeout", type=float, default=1.0)
    ap.add_argument("--total-sims", type=int, default=32)
    ap.add_argument("--num-candidates", type=int, default=4)
    ap.add_argument("--rollout-depth", type=int, default=15)
    ap.add_argument("--hard-deadline-ms", type=float, default=300.0)
    ap.add_argument("--exp3-eta", type=float, default=0.3)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--out", default="runs/ab_exp3_vs_ucb.json")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== AB Exp3 vs UCB at sim-move root ===")
    print(f"games={args.games}  step_timeout={args.step_timeout}  "
          f"total_sims={args.total_sims}  num_cand={args.num_candidates}  "
          f"rollout_depth={args.rollout_depth}  "
          f"hard_deadline_ms={args.hard_deadline_ms}  eta={args.exp3_eta}")
    print(f"out={out_path}\n")

    rows: List[GameRow] = []
    t0 = time.perf_counter()
    for i in range(args.games):
        seed = args.seed_base + i
        exp3_seat = i % 2
        row = play_one(
            seed=seed, exp3_seat=exp3_seat,
            step_timeout=args.step_timeout,
            total_sims=args.total_sims,
            num_cand=args.num_candidates,
            rollout_depth=args.rollout_depth,
            hard_deadline_ms=args.hard_deadline_ms,
            exp3_eta=args.exp3_eta,
        )
        row.game_idx = i
        rows.append(row)

    wall_total = time.perf_counter() - t0

    wins = sum(1 for r in rows if r.reward_exp3 == +1)
    losses = sum(1 for r in rows if r.reward_exp3 == -1)
    draws = sum(1 for r in rows if r.reward_exp3 == 0)
    n = wins + losses + draws
    wr = (wins + 0.5 * draws) / max(1, n)
    # Elo delta against a uniform prior (K=16 equivalent): inferred
    # from the observed win rate.
    elo_delta = -400 * math.log10(1 / max(1e-9, wr) - 1) if 0 < wr < 1 else float("inf")

    # Per-seat breakdown — in case exp3 has a seat bias.
    wr_seat0 = sum(1 if r.reward_exp3 == 1 else (0.5 if r.reward_exp3 == 0 else 0)
                   for r in rows if r.exp3_seat == 0)
    n_seat0 = sum(1 for r in rows if r.exp3_seat == 0)
    wr_seat1 = sum(1 if r.reward_exp3 == 1 else (0.5 if r.reward_exp3 == 0 else 0)
                   for r in rows if r.exp3_seat == 1)
    n_seat1 = sum(1 for r in rows if r.exp3_seat == 1)

    # All turn times across all games, per variant.
    all_exp3_p50 = [r.p50_exp3_ms for r in rows if not math.isnan(r.p50_exp3_ms)]
    all_ucb_p50 = [r.p50_ucb_ms for r in rows if not math.isnan(r.p50_ucb_ms)]
    all_exp3_max = [r.max_exp3_ms for r in rows if not math.isnan(r.max_exp3_ms)]
    all_ucb_max = [r.max_ucb_ms for r in rows if not math.isnan(r.max_ucb_ms)]

    print(f"\n=== Summary ({n} games, {wall_total/60:.1f} min wall) ===")
    print(f"Exp3:  {wins}W-{losses}L-{draws}D  wr={wr:.3f}  Elo delta={elo_delta:+.1f}")
    print(f"  seat 0: wr={wr_seat0/max(1,n_seat0):.3f}  (n={n_seat0})")
    print(f"  seat 1: wr={wr_seat1/max(1,n_seat1):.3f}  (n={n_seat1})")
    print(f"Turn-time p50 median: exp3={statistics.median(all_exp3_p50):.0f}ms  "
          f"ucb={statistics.median(all_ucb_p50):.0f}ms")
    print(f"Turn-time max max:    exp3={max(all_exp3_max):.0f}ms  "
          f"ucb={max(all_ucb_max):.0f}ms")
    gate = "PASS" if wr >= 0.55 else ("TIE" if 0.45 <= wr < 0.55 else "FAIL")
    print(f"\nGate @ 0.55: {gate}  (wr={wr:.3f})")

    out_payload = {
        "args": vars(args),
        "wall_total_s": wall_total,
        "games": [asdict(r) for r in rows],
        "summary": {
            "wins": wins, "losses": losses, "draws": draws,
            "n_games": n, "win_rate": wr, "elo_delta": elo_delta,
            "seat0_wr": wr_seat0 / max(1, n_seat0), "seat0_n": n_seat0,
            "seat1_wr": wr_seat1 / max(1, n_seat1), "seat1_n": n_seat1,
            "gate_055_pass": wr >= 0.55,
            "turn_time_p50_median_exp3_ms": statistics.median(all_exp3_p50),
            "turn_time_p50_median_ucb_ms": statistics.median(all_ucb_p50),
            "turn_time_max_max_exp3_ms": max(all_exp3_max),
            "turn_time_max_max_ucb_ms": max(all_ucb_max),
        },
    }
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"Written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
