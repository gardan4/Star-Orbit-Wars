"""Terminal dashboard for training/tuning/H2H runs.

Auto-detects what's running by scanning ``runs/`` for known patterns
and pretty-prints a refreshing view in your terminal. ANSI escape
codes only — no external deps. Refreshes every 5 seconds.

Tracks:

* **BC warm-start** — most recent ``runs/bc_warmstart_*.log`` (default,
  small, cpu, etc.). Reports current epoch, val-acc trajectory, last
  best, time-per-epoch.
* **TuRBO** — most recent ``runs/turbo_v?_*.log``. Reports current
  trial, best_so_far trajectory, recent trials, average wall.
* **Kaggle ladder** — most recent ``runs/v11_ladder_poll_*.log``.
  Reports latest ladder snapshot for v11/v9/v8.
* **In-flight H2H** — bundled-h2h logs in ``runs/`` matching
  ``*_h2h_*.log`` and not yet closed (final summary not present).

Run:
  $env:PYTHONPATH="src;."; .venv\\Scripts\\python.exe tools\\dashboard.py
  # Press Ctrl+C to exit.

Or one-shot:
  $env:PYTHONPATH="src;."; .venv\\Scripts\\python.exe tools\\dashboard.py --once
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
RUNS = _ROOT / "runs"

# ANSI codes — keep portable on modern Windows terminals (Windows
# Terminal + recent PowerShell understand them; legacy cmd.exe may
# echo them, but the dashboard still reads).
ANSI_CLEAR = "\033[2J\033[H"
ANSI_HOME = "\033[H"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_RED = "\033[31m"


def _newest(paths: Iterable[Path]) -> Optional[Path]:
    paths = list(paths)
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _read_tail(path: Path, n: int = 200) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return lines[-n:]
    except FileNotFoundError:
        return []


# ---------------------------------------------------------------------------
# BC warm-start panel
# ---------------------------------------------------------------------------


_BC_EPOCH_RE = re.compile(
    r"epoch\s+(\d+)/(\d+)\s+\*?\s*lr=([0-9.eE+-]+)\s+tr_loss=([0-9.]+)\s+"
    r"tr_acc=([0-9.]+)\s+va_loss=([0-9.]+)\s+va_acc=([0-9.]+)\s+best_va=([0-9.]+)"
)


def render_bc(width: int) -> List[str]:
    """Return rendered lines for the BC panel."""
    candidates = sorted(RUNS.glob("bc_warmstart*.log"))
    log = _newest(candidates)
    out: List[str] = []
    title = f"{ANSI_BOLD}{ANSI_GREEN}== BC warm-start =={ANSI_RESET}"
    if log is None:
        return [title, "  (no bc_warmstart_*.log found)"]
    out.append(f"{title}  {ANSI_DIM}{log.name}{ANSI_RESET}")

    lines = _read_tail(log, 400)
    epochs: List[Tuple[int, int, float, float, float]] = []  # (i, total, tr_acc, va_acc, best_va)
    model_params: Optional[int] = None
    device = "?"
    for line in lines:
        m = _BC_EPOCH_RE.search(line)
        if m:
            i = int(m.group(1)); tot = int(m.group(2))
            tr_acc = float(m.group(5)); va_acc = float(m.group(7))
            best = float(m.group(8))
            epochs.append((i, tot, tr_acc, va_acc, best))
        elif "model params:" in line:
            try:
                model_params = int(line.split("model params:")[1].strip().replace(",", ""))
            except Exception:
                pass
        elif "device=" in line and "device=cuda" in line.replace(" ", ""):
            device = "cuda"
        elif "device=" in line and "device=cpu" in line.replace(" ", ""):
            device = "cpu"

    if model_params is not None:
        out.append(f"  model: {model_params:,} params  device={device}")
    if not epochs:
        out.append(f"  {ANSI_YELLOW}(no epoch logged yet — training in setup or first epoch){ANSI_RESET}")
        return out

    last = epochs[-1]
    out.append(
        f"  epoch {last[0]:>2}/{last[1]:<2}  "
        f"tr_acc={last[2]:.3f}  "
        f"va_acc={last[3]:.3f}  "
        f"{ANSI_BOLD}best_va={last[4]:.3f}{ANSI_RESET}"
    )
    # Sparkline of va_acc trajectory.
    spark = " ".join(f"{e[3]:.3f}" for e in epochs)
    out.append(f"  trajectory: {ANSI_CYAN}{spark}{ANSI_RESET}")

    # Time per epoch from log mtime spacing — ~estimate.
    if len(epochs) >= 2:
        # Use file mtime as upper bound; can't get per-line timestamps
        # without instrumenting. Show last-epoch wall delta from log mtime
        # vs file ctime.
        mt = log.stat().st_mtime
        # Approximate: total wall / n epochs
        try:
            ct = log.stat().st_ctime
            total = mt - ct
            avg = total / max(1, len(epochs))
            out.append(f"  ~ avg wall: {avg/60:.1f} min/epoch (rough)")
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# TuRBO panel
# ---------------------------------------------------------------------------


_TURBO_TRIAL_RE = re.compile(
    r"\[t=\s*(\d+)\]\s+win=([0-9.]+)\s+hard=([0-9.]+)\s+n=(\d+)\s+([0-9.]+)s\s+best_so_far=([0-9.]+)"
)


def render_turbo(width: int) -> List[str]:
    candidates = sorted(RUNS.glob("turbo_v*_*.log"))
    log = _newest(candidates)
    out: List[str] = []
    title = f"{ANSI_BOLD}{ANSI_MAGENTA}== TuRBO weight tuning =={ANSI_RESET}"
    if log is None:
        return [title, "  (no turbo_v*_*.log found)"]
    out.append(f"{title}  {ANSI_DIM}{log.name}{ANSI_RESET}")

    lines = _read_tail(log, 1000)
    trials: List[Tuple[int, float, float, int, float, float]] = []
    for line in lines:
        m = _TURBO_TRIAL_RE.search(line)
        if m:
            t = int(m.group(1)); win = float(m.group(2)); hard = float(m.group(3))
            n = int(m.group(4)); wall = float(m.group(5)); best = float(m.group(6))
            trials.append((t, win, hard, n, wall, best))

    if not trials:
        out.append("  (no trials closed yet)")
        return out

    last = trials[-1]
    avg_wall = sum(t[4] for t in trials) / len(trials)
    out.append(
        f"  trial {last[0]:>2}/60  "
        f"win={last[1]:.3f}  "
        f"{ANSI_BOLD}best_so_far={last[5]:.3f}{ANSI_RESET}  "
        f"avg_wall={avg_wall:.0f}s"
    )
    # Recent 6 trials, color-coded vs best.
    recent = trials[-8:]
    rec_strs: List[str] = []
    for t in recent:
        c = ANSI_GREEN if t[1] >= last[5] else (ANSI_YELLOW if t[1] >= 0.7 else ANSI_RED)
        rec_strs.append(f"{c}{t[1]:.2f}{ANSI_RESET}")
    out.append(f"  recent: {' '.join(rec_strs)}")

    # Best-so-far trajectory at every improvement.
    best_evolutions: List[Tuple[int, float]] = []
    bsf = -1.0
    for t in trials:
        if t[5] > bsf:
            best_evolutions.append((t[0], t[5]))
            bsf = t[5]
    bsf_strs = [f"t{e[0]}={ANSI_BOLD}{e[1]:.3f}{ANSI_RESET}" for e in best_evolutions]
    out.append(f"  best @: {' '.join(bsf_strs)}")
    return out


# ---------------------------------------------------------------------------
# Ladder panel
# ---------------------------------------------------------------------------


_LADDER_RE = re.compile(
    r"submission\.py\s+([0-9-]+ [0-9:.]+).*?(v\d+:.*?)\s+SubmissionStatus\.COMPLETE\s+([0-9.]+)"
)


def render_ladder(width: int) -> List[str]:
    candidates = sorted(RUNS.glob("v*_ladder_poll*.log"))
    log = _newest(candidates)
    out: List[str] = []
    title = f"{ANSI_BOLD}{ANSI_BLUE}== Kaggle ladder =={ANSI_RESET}"
    if log is None:
        return [title, "  (no ladder poll log)"]
    out.append(f"{title}  {ANSI_DIM}{log.name}{ANSI_RESET}")

    lines = _read_tail(log, 200)
    # Last poll block ends at a "=== poll" header for the next one or EOF.
    snapshot: List[Tuple[str, float]] = []  # (label, score)
    last_poll_label = ""
    for line in lines[::-1]:
        if line.startswith("=== poll"):
            last_poll_label = line.strip()
            break
        m = _LADDER_RE.search(line)
        if m:
            ts = m.group(1).split()[0]
            desc = m.group(2)[:30].strip()
            score = float(m.group(3))
            # Pull version from desc prefix "vN:"
            ver = desc.split(":")[0] if ":" in desc else desc[:6]
            snapshot.append((ver, score))
    if last_poll_label:
        out.append(f"  {ANSI_DIM}{last_poll_label}{ANSI_RESET}")
    if not snapshot:
        out.append("  (no submissions parsed yet)")
        return out
    # Reverse to top-down order.
    snapshot = snapshot[::-1]
    # Highlight the top one.
    for i, (v, s) in enumerate(snapshot[:5]):
        color = ANSI_GREEN if i == 0 else ANSI_RESET
        marker = "*" if i == 0 else " "
        out.append(f"  {marker} {color}{v:<6}{ANSI_RESET}  {s:>7.1f}")
    return out


# ---------------------------------------------------------------------------
# H2H in-flight panel
# ---------------------------------------------------------------------------


_H2H_GAME_RE = re.compile(
    r"g(\d+):\s+seed=\d+\s+(\S+)=seat\d,\s+(\S+)=seat\d\s+steps=(\d+)\s+(\S+)\s+(WIN|LOSS|TIE)"
)
_H2H_SUMMARY_RE = re.compile(r"--- Summary \((\d+) games, ([0-9.]+)s\) ---")
_H2H_HEADER_RE = re.compile(r"Round-robin:\s+(\d+) bundles, (\d+) games/pair")


def render_h2h(width: int) -> List[str]:
    """Show every in-flight H2H log we can find."""
    out: List[str] = []
    title = f"{ANSI_BOLD}{ANSI_YELLOW}== Bundled H2H =={ANSI_RESET}"
    out.append(title)
    h2h_logs = sorted(RUNS.glob("*h2h*.log"), key=lambda p: -p.stat().st_mtime)[:3]
    if not h2h_logs:
        out.append("  (no h2h logs)")
        return out
    for log in h2h_logs:
        lines = _read_tail(log, 200)
        games: List[Tuple[int, str, str, int, str, str]] = []
        n_games_planned = None
        summary = None
        for line in lines:
            mh = _H2H_HEADER_RE.search(line)
            if mh:
                n_games_planned = int(mh.group(2))
            mg = _H2H_GAME_RE.search(line)
            if mg:
                games.append((
                    int(mg.group(1)), mg.group(2), mg.group(3),
                    int(mg.group(4)), mg.group(5), mg.group(6),
                ))
            ms = _H2H_SUMMARY_RE.search(line)
            if ms:
                summary = ms.group(0)
        # Mark closed if "Elo ranking" or summary present.
        closed = summary is not None or any("Elo ranking" in l for l in lines)
        status = f"{ANSI_DIM}DONE{ANSI_RESET}" if closed else f"{ANSI_GREEN}LIVE{ANSI_RESET}"
        head = f"  {status}  {log.name}  ({len(games)}/{n_games_planned or '?'} games)"
        out.append(head)
        if games and not closed:
            # Tally. Game line format:
            #   gN: seed=S botA=seat0, botB=seat1 steps=K NAMED RESULT
            # where NAMED is the bot that the RESULT (WIN/LOSS/TIE) refers
            # to. So `v12_e5 LOSS` means v12_e5 LOST → opponent won.
            bots = sorted({g[1] for g in games} | {g[2] for g in games})
            wl: dict = {b: [0, 0, 0] for b in bots}  # W, L, T
            for g in games:
                named = g[4]  # bot that the WIN/LOSS/TIE describes
                outcome = g[5]
                opponent = next((b for b in bots if b != named), None)
                if outcome == "TIE":
                    for b in bots:
                        wl[b][2] += 1
                elif outcome == "WIN":
                    if named in wl:
                        wl[named][0] += 1
                    if opponent in wl:
                        wl[opponent][1] += 1
                elif outcome == "LOSS":
                    if named in wl:
                        wl[named][1] += 1
                    if opponent in wl:
                        wl[opponent][0] += 1
            # Print first bot only (it's the candidate)
            for b in bots[:2]:
                w, l, t = wl[b]
                out.append(f"     {b:<14}  W{w:<2} L{l:<2} T{t:<2}")
        if closed and summary:
            out.append(f"     {ANSI_DIM}{summary}{ANSI_RESET}")
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def render_all() -> str:
    width = 80
    parts: List[str] = []
    parts.append(f"{ANSI_BOLD}Orbit Wars training dashboard{ANSI_RESET}  "
                 f"{ANSI_DIM}{time.strftime('%Y-%m-%d %H:%M:%S')}{ANSI_RESET}")
    parts.append("")
    for renderer in (render_bc, render_turbo, render_ladder, render_h2h):
        parts.extend(renderer(width))
        parts.append("")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="Print one snapshot and exit")
    ap.add_argument("--interval", type=int, default=5,
                    help="Refresh interval in seconds (default 5)")
    args = ap.parse_args()

    if args.once:
        print(render_all())
        return 0

    # Enable ANSI on Windows.
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

    sys.stdout.write(ANSI_CLEAR)
    try:
        while True:
            sys.stdout.write(ANSI_HOME)
            sys.stdout.write(render_all())
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
