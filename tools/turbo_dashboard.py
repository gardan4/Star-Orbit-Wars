"""Live text dashboard for a running TuRBO / Ax tuning job.

Reads ``runs/<name>.json`` every N seconds and pretty-prints:
  * trials completed / total, best-so-far, wall time used, ETA,
  * per-trial win_rate (with the best one marked),
  * per-opponent breakdown for the current best trial,
  * the current-best parameter diffs against the defaults,
  * baseline reference line (defaults vs w2 pool) if available.

Usage:
    python tools/turbo_dashboard.py
    python tools/turbo_dashboard.py --path runs/turbo_v2_20260424.json \
        --baseline runs/baseline_w2_defaults_20260424.json --refresh 20

Designed to be pointed at by ``Start-Process`` and left open in its own
PowerShell window. It clears the screen each refresh so a single pane
always shows current state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    m = s / 60
    if m < 60:
        return f"{m:.1f}m"
    h = m / 60
    return f"{h:.1f}h"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text()
        return json.loads(text)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        # File is being written — try again next tick.
        return None


def _format_weights_diff(weights: Dict[str, float]) -> str:
    """Show top-5 largest-magnitude relative differences vs defaults."""
    diffs = []
    for k, v in weights.items():
        default = HEURISTIC_WEIGHTS.get(k)
        if default is None:
            continue
        # Relative diff (guarded against division by zero).
        if abs(default) < 1e-6:
            rel = abs(v - default)
        else:
            rel = (v - default) / abs(default)
        diffs.append((abs(rel), k, default, v, rel))
    diffs.sort(reverse=True)
    lines = []
    for _, k, d, v, rel in diffs[:5]:
        sign = "+" if rel >= 0 else "-"
        lines.append(
            f"    {k:<26} {d:>7.3f} -> {v:>7.3f}  ({sign}{abs(rel)*100:.0f}%)"
        )
    return "\n".join(lines)


def _render(
    job: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
    path: Path,
) -> str:
    n_done = len(job.get("trials", []))
    n_total = int(job.get("n_trials", 0))
    started = job.get("started_at", 0.0)
    now = time.time()
    wall_used = now - started if started else 0.0
    best_wr = job.get("best_win_rate", float("-inf"))
    best_i = job.get("best_index", -1)

    lines = []
    lines.append(f"=== TuRBO dashboard — {path.name}  ({time.strftime('%H:%M:%S')}) ===")
    lines.append(
        f"strategy={job.get('strategy','?'):<10} "
        f"trials={n_done}/{n_total}  "
        f"wall={_fmt_secs(wall_used)}"
    )
    if n_done > 0:
        avg_trial_sec = wall_used / n_done
        remaining = (n_total - n_done) * avg_trial_sec
        lines.append(
            f"avg/trial={_fmt_secs(avg_trial_sec)}  "
            f"ETA={_fmt_secs(remaining)}  "
            f"(finishes ~{time.strftime('%H:%M', time.localtime(now + remaining))})"
        )
    if baseline is not None:
        bwr = baseline.get("win_rate")
        bn = baseline.get("n_games")
        lines.append(
            f"BASELINE (defaults vs w2 pool): win_rate={bwr:.3f}  n={bn}  "
            f"-> bar to beat >= 0.74  (0.667 + 7pp SE)"
        )
    lines.append("")

    # Per-trial bars
    if n_done > 0:
        lines.append("Per-trial win_rate:")
        for t in job["trials"]:
            idx = t["trial"]
            wr = t["win_rate"]
            wall = t.get("fitness_wall_seconds", 0.0)
            bar_width = int(wr * 40)
            bar = "#" * bar_width + "." * (40 - bar_width)
            mark = " *" if idx == best_i else "  "
            lines.append(
                f"  t{idx:>2}{mark} [{bar}] {wr:.3f}  wall={_fmt_secs(wall)}"
            )
        lines.append("")

    # Best per-opp breakdown
    if best_i >= 0 and n_done > 0:
        best_trial = next((t for t in job["trials"] if t["trial"] == best_i), None)
        if best_trial is not None:
            lines.append(f"Best trial {best_i} by-opponent (vs w2 pool):")
            by_opp = best_trial.get("by_opponent", {})
            for name in sorted(by_opp.keys()):
                wr, n = by_opp[name]
                wins = int(wr * n)
                lines.append(f"  {name:<15} {wr:.3f}  ({wins}/{n})")
            lines.append("")
            lines.append(f"Best trial {best_i} weight diffs vs defaults (top 5):")
            lines.append(_format_weights_diff(best_trial["weights"]))
            lines.append("")

    lines.append(
        "Refresh: every N seconds (Ctrl-C to stop). "
        "JSON source: " + str(path)
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="runs/turbo_v2_20260424.json")
    ap.add_argument(
        "--baseline", default="runs/baseline_w2_defaults_20260424.json"
    )
    ap.add_argument("--refresh", type=int, default=30)
    args = ap.parse_args()

    path = Path(args.path)
    baseline_path = Path(args.baseline) if args.baseline else None

    print(f"Watching {path} (refresh every {args.refresh}s). Ctrl-C to stop.")
    while True:
        job = _read_json(path)
        baseline = (
            _read_json(baseline_path)
            if baseline_path is not None
            else None
        )
        if job is None:
            # Clear screen on Windows.
            os.system("cls")
            print(f"[{time.strftime('%H:%M:%S')}] waiting for {path} ...")
        else:
            os.system("cls")
            print(_render(job, baseline, path))
        try:
            time.sleep(args.refresh)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
