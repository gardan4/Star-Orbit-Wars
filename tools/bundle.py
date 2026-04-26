"""Single-file submission packager.

Kaggle's orbit_wars submission is ONE Python file exposing a
`def agent(obs, config=None) -> list` callable. This tool inlines our
`orbitwars` package code into such a file.

Design decisions:
  * We inline by module, in explicit dependency order — simpler than AST
    rewriting and keeps diagnostics readable in Kaggle's log (tracebacks
    still point to meaningful source lines).
  * Any `from orbitwars.xxx import y` import that references an inlined
    module is replaced with a harmless pass-through (the names are already
    in scope).
  * External imports (numpy, math, random, ...) are preserved as-is. We do
    NOT inline anything from stdlib or site-packages.
  * Emit a `def agent(obs, config=None):` at the end wrapping
    `HeuristicAgent().as_kaggle_agent()`.

Usage:
    python -m tools.bundle --bot heuristic --out submissions/heuristic_v1.py
    python -m tools.bundle --bot noop --out submissions/noop.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src" / "orbitwars"


# Which bots we know how to package (name -> (module path, agent factory line))
#
# Module order matters: each module may reference names defined in earlier
# modules (those imports are stripped by the bundler). The listed order
# must satisfy dependency topo-sort.
BOT_RECIPES: Dict[str, Tuple[List[str], str]] = {
    "noop": (
        ["bots.base"],
        "agent = NoOpAgent().as_kaggle_agent()",
    ),
    "random": (
        ["bots.base"],
        "agent = RandomAgent(seed=0).as_kaggle_agent()",
    ),
    "heuristic": (
        ["engine.intercept", "bots.base", "bots.heuristic"],
        "agent = HeuristicAgent().as_kaggle_agent()",
    ),
    # Path B bot. Currently (W2/W3) equivalent to heuristic at the wire
    # level because `anchor_improvement_margin=2.0` locks the heuristic
    # floor — search still runs but never overrides the anchor. Keep
    # this recipe so the moment we flip search net-positive (W4-5 neural
    # prior) we can ship it with one bundle command. Bundle size: ~3k
    # lines; stays well under Kaggle's per-file limit.
    "mcts_bot": (
        [
            "engine.intercept",
            "bots.base",
            "bots.heuristic",
            "bots.fast_rollout",  # referenced by gumbel_search when rollout_policy="fast"
            "engine.fast_engine",
            "mcts.bokr_widen",    # provides BOKRKernelSelector used by mcts.actions
            "mcts.actions",
            "mcts.gumbel_search",
            "opponent.archetypes",
            "opponent.bayes",
            "bots.mcts_bot",
        ],
        "agent = MCTSAgent(rng_seed=0).as_kaggle_agent()",
    ),
}


def _strip_orbitwars_imports(src: str) -> str:
    """Remove `from orbitwars.xxx import ...` (including parenthesized
    multi-line import blocks). Those names are already defined in the
    bundled file after inlining."""
    lines = src.splitlines()
    out_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("from orbitwars.") or stripped.startswith("import orbitwars."):
            # Multi-line case: `from orbitwars.x import (` spans to matching `)`.
            if "(" in line and ")" not in line:
                # Skip until the line containing the matching `)`.
                while i < len(lines) and ")" not in lines[i]:
                    i += 1
                # Skip the closing line too
                i += 1
                continue
            # Single-line case: skip this line only.
            i += 1
            continue
        out_lines.append(line)
        i += 1
    return "\n".join(out_lines)


def _strip_leading_docstring_module_header(src: str) -> str:
    """Drop the module-level docstring and `from __future__` line if present —
    they don't belong mid-file after the first module."""
    # This is a soft heuristic; simplest approach is to keep them. A bundled
    # file with multiple `from __future__` lines is fine in Python 3.13.
    return src


def _read_module(rel_path: str) -> str:
    """rel_path e.g. 'bots.base' -> reads src/orbitwars/bots/base.py"""
    parts = rel_path.split(".")
    path = SRC_ROOT.joinpath(*parts).with_suffix(".py")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


HEADER = """\
# Auto-generated Orbit Wars submission. Do not edit by hand.
# Built by tools/bundle.py on {timestamp}.
# Bot: {bot_name}
#
# Single-file submission: Kaggle will import this and call agent(obs, cfg).
from __future__ import annotations

"""


def bundle_bot(
    bot_name: str,
    out_path: Path,
    weights_override: Dict[str, float] | None = None,
    sim_move_variant: str | None = None,
    exp3_eta: float | None = None,
    rollout_policy: str | None = None,
    anchor_margin: float | None = None,
    use_opponent_model: bool | None = None,
    nn_checkpoint: Path | None = None,
    nn_temperature: float | None = None,
    nn_hold_neutral_prob: float | None = None,
    use_macros: bool | None = None,
    nn_dataset_path: str | None = None,
    total_sims: int | None = None,
    hard_deadline_ms: float | None = None,
) -> None:
    if bot_name not in BOT_RECIPES:
        raise SystemExit(f"Unknown bot '{bot_name}'. Known: {list(BOT_RECIPES)}")
    modules, agent_factory = BOT_RECIPES[bot_name]
    # If --nn-checkpoint is set, splice the NN modules into the inline
    # list AFTER bots.heuristic (which defines PlanetMove via mcts.actions
    # transitively) but BEFORE bots.mcts_bot (which constructs the agent).
    # The ConvPolicy import path nn.conv_policy → no orbitwars sibling
    # imports inside it (only torch), so order with respect to other
    # orbitwars modules doesn't matter beyond "before mcts_bot".
    if nn_checkpoint is not None and nn_dataset_path is not None:
        raise SystemExit(
            "Use either --nn-checkpoint (base64-inline) or "
            "--nn-dataset-path (read from Kaggle Dataset at runtime), not both."
        )

    if nn_checkpoint is not None or nn_dataset_path is not None:
        if bot_name != "mcts_bot":
            raise SystemExit(
                f"--nn-checkpoint / --nn-dataset-path only apply to "
                f"bot=mcts_bot (got {bot_name!r})"
            )
        modules = list(modules)
        # Insert just before bots.mcts_bot (which is the last entry).
        insert_at = modules.index("bots.mcts_bot")
        modules.insert(insert_at, "nn.conv_policy")
        modules.insert(insert_at + 1, "nn.nn_prior")

    if use_macros:
        if bot_name != "mcts_bot":
            raise SystemExit(
                f"--use-macros only applies to bot=mcts_bot (got {bot_name!r})"
            )
        modules = list(modules)
        # mcts.macros has no orbitwars siblings except the modules already
        # in the recipe. Insert before bots.mcts_bot.
        insert_at = modules.index("bots.mcts_bot")
        if "mcts.macros" not in modules:
            modules.insert(insert_at, "mcts.macros")

    parts: List[str] = [HEADER.format(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), bot_name=bot_name)]
    seen_future: List[str] = []

    for mod in modules:
        src = _read_module(mod)
        src = _strip_orbitwars_imports(src)
        # Consolidate multiple `from __future__ import annotations` into one
        # at the top; python requires them to be at the start of file.
        src = re.sub(r"^from __future__ import annotations\s*$", "", src, flags=re.MULTILINE)
        parts.append(f"# --- inlined: orbitwars/{mod.replace('.', '/')}.py ---\n")
        parts.append(src.rstrip() + "\n")
        parts.append("\n")

    # Optional weights override — emitted BEFORE the agent factory so
    # the dict is patched in place before HeuristicAgent reads it. We
    # update-in-place rather than reassign so any earlier reference
    # (e.g. archetype weight builders in opponent/archetypes.py) still
    # points at the live dict.
    if weights_override:
        parts.append("\n# --- tuned weights override ---\n")
        parts.append("# Applied by tools/bundle.py --weights-json at build time.\n")
        # Emit as a literal dict so the bundle stays standalone (no JSON
        # loading at Kaggle boot — one less moving piece).
        items = ",\n    ".join(
            f"{k!r}: {float(v)!r}" for k, v in sorted(weights_override.items())
        )
        parts.append(
            f"HEURISTIC_WEIGHTS.update({{\n    {items},\n}})\n"
        )

    # Optional GumbelConfig overrides for the mcts_bot recipe. Each knob
    # below maps to a single attribute on a `_bundle_cfg = GumbelConfig()`
    # instance that we emit into the bundle, then we rewrite the agent
    # factory to use that cfg (and optionally flip `use_opponent_model`).
    # Knobs supported:
    #   * --sim-move-variant / --exp3-eta (W4 A/B: exp3 decoupled root)
    #   * --rollout-policy (W4 perf: fast vs heuristic rollouts)
    #   * --anchor-margin (relaxes anchor-lock for MCTS overrides)
    #   * --use-opponent-model=false (disables Bayes observe path)
    cfg_setters: List[str] = []  # each "_bundle_cfg.KEY = VAL" line
    factory_kwargs: List[str] = []  # extra MCTSAgent() kwargs beyond rng_seed

    # Validation + emission per knob.
    if sim_move_variant is not None:
        if bot_name != "mcts_bot":
            raise SystemExit(
                f"--sim-move-variant only applies to bot=mcts_bot (got {bot_name!r})"
            )
        if sim_move_variant not in ("ucb", "exp3"):
            raise SystemExit(
                f"--sim-move-variant must be 'ucb' or 'exp3' (got {sim_move_variant!r})"
            )
        cfg_setters.append(f"_bundle_cfg.sim_move_variant = {sim_move_variant!r}")
        eta_val = float(exp3_eta) if exp3_eta is not None else 0.3
        cfg_setters.append(f"_bundle_cfg.exp3_eta = {eta_val!r}")

    if rollout_policy is not None:
        if bot_name != "mcts_bot":
            raise SystemExit(
                f"--rollout-policy only applies to bot=mcts_bot (got {bot_name!r})"
            )
        if rollout_policy not in ("heuristic", "fast", "nn_value"):
            raise SystemExit(
                f"--rollout-policy must be 'heuristic'|'fast'|'nn_value' "
                f"(got {rollout_policy!r})"
            )
        if rollout_policy == "nn_value" and nn_checkpoint is None and nn_dataset_path is None:
            raise SystemExit(
                "--rollout-policy=nn_value requires --nn-checkpoint or "
                "--nn-dataset-path so the value head can be loaded; "
                "without one the search has no value_fn to call."
            )
        cfg_setters.append(f"_bundle_cfg.rollout_policy = {rollout_policy!r}")

    if anchor_margin is not None:
        if bot_name != "mcts_bot":
            raise SystemExit(
                f"--anchor-margin only applies to bot=mcts_bot (got {bot_name!r})"
            )
        cfg_setters.append(
            f"_bundle_cfg.anchor_improvement_margin = {float(anchor_margin)!r}"
        )

    if use_opponent_model is not None:
        if bot_name != "mcts_bot":
            raise SystemExit(
                f"--use-opponent-model only applies to bot=mcts_bot (got {bot_name!r})"
            )
        # Note: use_opponent_model is an MCTSAgent constructor arg, NOT a
        # GumbelConfig attribute. It goes into factory_kwargs, not cfg_setters.
        factory_kwargs.append(f"use_opponent_model={bool(use_opponent_model)!r}")

    if use_macros is not None:
        if bot_name != "mcts_bot":
            raise SystemExit(
                f"--use-macros only applies to bot=mcts_bot (got {bot_name!r})"
            )
        cfg_setters.append(f"_bundle_cfg.use_macros = {bool(use_macros)!r}")

    if total_sims is not None:
        if bot_name != "mcts_bot":
            raise SystemExit(
                f"--total-sims only applies to bot=mcts_bot (got {bot_name!r})"
            )
        cfg_setters.append(f"_bundle_cfg.total_sims = {int(total_sims)!r}")

    if hard_deadline_ms is not None:
        if bot_name != "mcts_bot":
            raise SystemExit(
                f"--hard-deadline-ms only applies to bot=mcts_bot (got {bot_name!r})"
            )
        cfg_setters.append(f"_bundle_cfg.hard_deadline_ms = {float(hard_deadline_ms)!r}")

    # NN-prior wiring. The .pt checkpoint is base64-inlined into the
    # bundled .py so the submission is single-file (no separate Kaggle
    # Dataset upload step). At ~1.8 MB fp32 the encoded text is ~2.4 MB,
    # comfortably under Kaggle's per-file caps. We emit a small bootstrap
    # block that decodes, runs torch.load on a BytesIO, constructs the
    # ConvPolicy, and builds a `_bundle_move_prior_fn` that's threaded
    # into the agent factory below.
    nn_bootstrap_lines: List[str] = []
    if nn_checkpoint is not None or nn_dataset_path is not None:
        temp = float(nn_temperature) if nn_temperature is not None else 1.0
        hold_p = float(nn_hold_neutral_prob) if nn_hold_neutral_prob is not None else 0.05

        load_lines: List[str] = []
        header_comment: str = ""
        if nn_checkpoint is not None:
            ckpt_path = Path(nn_checkpoint)
            if not ckpt_path.exists():
                raise SystemExit(f"--nn-checkpoint: file not found at {ckpt_path}")
            import base64 as _b64
            ckpt_bytes = ckpt_path.read_bytes()
            encoded = _b64.b64encode(ckpt_bytes).decode("ascii")
            wrapped = "\n".join(
                encoded[i : i + 76] for i in range(0, len(encoded), 76)
            )
            header_comment = "--nn-checkpoint (base64 inline)"
            load_lines = [
                "import base64 as _bundle_b64",
                "import io as _bundle_io",
                "import torch as _bundle_torch",
                "_BUNDLE_BC_CKPT_B64 = (",
                *(f"    {line!r}" for line in wrapped.split("\n")),
                ")",
                "_bundle_ckpt_bytes = _bundle_b64.b64decode(\"\".join(_BUNDLE_BC_CKPT_B64))",
                "_bundle_ckpt = _bundle_torch.load(",
                "    _bundle_io.BytesIO(_bundle_ckpt_bytes),",
                "    map_location=\"cpu\", weights_only=False,",
                ")",
            ]
        else:
            # Kaggle Dataset path: bundle reads the .pt at runtime from
            # /kaggle/input/<slug>/<file>.  This adds zero bytes to the
            # bundled .py and bypasses Kaggle's ~1-2 MB notebook-size cap.
            # Caller must also set kernel-metadata.json's `dataset_sources`
            # (see tools/build_kaggle_notebook.py --dataset-source).
            assert nn_dataset_path is not None
            ds_path = nn_dataset_path
            header_comment = f"--nn-dataset-path {ds_path}"
            load_lines = [
                "import torch as _bundle_torch",
                f"_BUNDLE_BC_CKPT_PATH = {ds_path!r}",
                "_bundle_ckpt = _bundle_torch.load(",
                "    _BUNDLE_BC_CKPT_PATH, map_location=\"cpu\", weights_only=False,",
                ")",
            ]

        nn_bootstrap_lines = [
            f"\n# --- NN prior bootstrap ({header_comment}) ---",
            *load_lines,
            "# Decode any quantized weights back to fp32 so the fp32",
            "# ConvPolicy module accepts the state_dict cleanly. fp16 halves",
            "# bundle size; int8_per_tensor_symmetric quarters it. Inference",
            "# precision is fp32 either way.",
            "def _bundle_upcast(sd, scales=None):",
            "    out = {}",
            "    for k, v in sd.items():",
            "        if v.dtype == torch.int8 and scales is not None and k in scales:",
            "            out[k] = v.float() * float(scales[k])",
            "        elif hasattr(v, 'is_floating_point') and v.is_floating_point():",
            "            out[k] = v.float()",
            "        else:",
            "            out[k] = v",
            "    return out",
            "_bundle_scales = _bundle_ckpt.get('_quant_scales')",
            "if 'model_state' in _bundle_ckpt and 'cfg' in _bundle_ckpt:",
            "    _bundle_cfg_nn = ConvPolicyCfg(**_bundle_ckpt['cfg'])",
            "    _bundle_model = ConvPolicy(_bundle_cfg_nn)",
            "    _bundle_model.load_state_dict(_bundle_upcast(_bundle_ckpt['model_state'], _bundle_scales))",
            "elif 'model_state_dict' in _bundle_ckpt:",
            "    _bundle_cfg_nn = ConvPolicyCfg()",
            "    _bundle_model = ConvPolicy(_bundle_cfg_nn)",
            "    _bundle_model.load_state_dict(_bundle_upcast(_bundle_ckpt['model_state_dict']))",
            "else:",
            "    raise RuntimeError('bundle: NN checkpoint has unrecognized keys')",
            "_bundle_model.eval()",
            "_bundle_move_prior_fn = make_nn_prior_fn(",
            "    _bundle_model, _bundle_cfg_nn,",
            f"    hold_neutral_prob={hold_p!r}, temperature={temp!r},",
            ")",
            "# Build value_fn from the same model. The value head is only",
            "# used when GumbelConfig.rollout_policy='nn_value'; building",
            "# the closure unconditionally costs ~0 bytes (just a closure)",
            "# and lets the same bundle support both rollout modes.",
            "from orbitwars.nn.nn_value import make_nn_value_fn",
            "_bundle_value_fn = make_nn_value_fn(_bundle_model, _bundle_cfg_nn)",
            "del _bundle_ckpt  # free RAM after model is built",
        ]
        factory_kwargs.append("move_prior_fn=_bundle_move_prior_fn")
        factory_kwargs.append("value_fn=_bundle_value_fn")

    if nn_bootstrap_lines:
        # Emit the NN bootstrap BEFORE the GumbelConfig / factory block so
        # `_bundle_move_prior_fn` is in scope when the factory references it.
        parts.append("\n".join(nn_bootstrap_lines) + "\n")

    if cfg_setters or factory_kwargs:
        parts.append("\n# --- GumbelConfig / MCTSAgent overrides ---\n")
        parts.append("# Applied by tools/bundle.py at build time.\n")
        if cfg_setters:
            parts.append("_bundle_cfg = GumbelConfig()\n")
            for line in cfg_setters:
                parts.append(line + "\n")
        # Rewrite the factory to thread the cfg (if any) and extra kwargs.
        new_kwargs: List[str] = []
        if cfg_setters:
            new_kwargs.append("gumbel_cfg=_bundle_cfg")
        new_kwargs.append("rng_seed=0")
        new_kwargs.extend(factory_kwargs)
        replacement = f"MCTSAgent({', '.join(new_kwargs)})"
        new_factory = agent_factory.replace("MCTSAgent(rng_seed=0)", replacement)
        if new_factory == agent_factory:
            raise SystemExit(
                "bundle.py: couldn't find 'MCTSAgent(rng_seed=0)' in "
                f"agent factory {agent_factory!r} to inject overrides"
            )
        agent_factory = new_factory

    # Agent entry point
    parts.append("\n# --- agent entry point ---\n")
    parts.append(f"{agent_factory}\n")

    content = "\n".join(parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def _smoke_test(out_path: Path) -> None:
    """Import the bundled file in a subprocess and run one game vs noop."""
    import subprocess
    code = f"""
import sys, traceback
sys.path.insert(0, r'{out_path.parent}')
mod_name = '{out_path.stem}'
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, r'{out_path}')
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m  # dataclasses need the module in sys.modules
    spec.loader.exec_module(m)
    assert callable(m.agent), 'agent not callable'
    from kaggle_environments import make
    env = make('orbit_wars', configuration={{'actTimeout': 60}}, debug=False)
    env.run([m.agent, 'random'])
    rewards = [s.reward for s in env.state]
    steps = env.state[0].observation.step
    print(f'OK rewards={{rewards}} steps={{steps}}')
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise SystemExit("Smoke test failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", required=True, choices=list(BOT_RECIPES))
    ap.add_argument("--out", required=True, help="Output .py path")
    ap.add_argument(
        "--weights-json",
        default=None,
        help=(
            "Path to a JSON file with tuned heuristic weights. Accepts "
            "either a top-level dict of {weight_name: value} OR one of the "
            "TuRBO output shapes ({'best_weights': {...}} / "
            "{'best_point': {...}} / {'best': {...}}). Emits a "
            "HEURISTIC_WEIGHTS.update({...}) line after the heuristic "
            "module so the bundled agent uses the tuned values."
        ),
    )
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run a single game against 'random' to verify the bundle works")
    ap.add_argument(
        "--sim-move-variant",
        default=None,
        choices=["ucb", "exp3"],
        help=(
            "Override GumbelConfig.sim_move_variant at bundle time. Only "
            "applies to bot=mcts_bot. Default behavior (when flag is "
            "omitted) matches the source tree default ('ucb'). Use 'exp3' "
            "to ship the Exp3 decoupled-root variant; requires the W4 "
            "A/B gate to have passed."
        ),
    )
    ap.add_argument(
        "--exp3-eta",
        type=float,
        default=None,
        help=(
            "Learning rate for Exp3 softmax. Only used when "
            "--sim-move-variant=exp3. Defaults to 0.3 (the sim_move.py "
            "module default)."
        ),
    )
    ap.add_argument(
        "--rollout-policy",
        default=None,
        choices=["heuristic", "fast", "nn_value"],
        help=(
            "Override GumbelConfig.rollout_policy at bundle time. "
            "nn_value: skip rollouts, query NN value head 1 ply ahead "
            "(requires --nn-checkpoint or --nn-dataset-path). "
            "'fast' uses FastRolloutAgent inside rollouts (~13× more "
            "sims/turn) but is only useful if combined with a relaxed "
            "--anchor-margin — under the shipped margin=2.0 the wire "
            "action is margin-locked and rollout count is irrelevant."
        ),
    )
    ap.add_argument(
        "--anchor-margin",
        type=float,
        default=None,
        help=(
            "Override GumbelConfig.anchor_improvement_margin. Default "
            "2.0 (effectively 'always play the heuristic'). Lower "
            "values (0.5, 0.0) let MCTS override the heuristic when "
            "the search Q-value exceeds the anchor by at least that "
            "margin. Only safe to lower alongside --rollout-policy=fast "
            "(need sim budget above the rollout noise floor)."
        ),
    )
    ap.add_argument(
        "--use-opponent-model",
        default=None,
        choices=["true", "false"],
        help=(
            "Override MCTSAgent use_opponent_model. Default True. "
            "Setting 'false' disables the Bayesian archetype posterior "
            "(saves ~27 ms/turn observe cost; use for clean MCTS-only "
            "experiments where opp_policy_override is a confound)."
        ),
    )
    ap.add_argument(
        "--nn-checkpoint",
        default=None,
        help=(
            "Path to a BC-warmstart .pt checkpoint produced by "
            "tools/bc_warmstart.py. When set, the conv_policy + nn_prior "
            "modules are inlined into the bundle, the checkpoint is "
            "base64-encoded into the .py file, and the agent is wired "
            "with a ConvPolicy-derived move_prior_fn that overwrites the "
            "heuristic priors at the MCTS root. Only applies to "
            "bot=mcts_bot. Bundle size grows by ~1.4× the checkpoint "
            "size (~2.4 MB for the default ConvPolicyCfg). For larger "
            "checkpoints that exceed Kaggle's notebook size limit, use "
            "--nn-dataset-path instead."
        ),
    )
    ap.add_argument(
        "--nn-dataset-path",
        default=None,
        help=(
            "Runtime path to the BC checkpoint inside a mounted Kaggle "
            "Dataset (e.g. '/kaggle/input/orbit-wars-bc-v2/bc_warmstart_v2.pt'). "
            "Bundle reads it via torch.load at startup instead of "
            "base64-decoding inline. Adds zero bytes to the .py, but the "
            "Kaggle Notebook MUST list the dataset slug under "
            "kernel-metadata.json's `dataset_sources` (use "
            "tools/build_kaggle_notebook.py --dataset-source)."
        ),
    )
    ap.add_argument(
        "--nn-temperature",
        type=float,
        default=None,
        help=(
            "Softmax temperature for NN prior. Default 1.0. >1 flattens "
            "(more exploration), <1 sharpens (more confidence in NN). "
            "Only used when --nn-checkpoint is set."
        ),
    )
    ap.add_argument(
        "--use-macros",
        default=None,
        choices=["true", "false"],
        help=(
            "Inject the 4-macro library (HOLD-all, mass-attack-nearest, "
            "reinforce-weakest, retreat-to-largest) as additional root "
            "candidates alongside the heuristic anchor. Plan §6.4. "
            "Only applies to bot=mcts_bot. Off by default for "
            "bit-identity with the v12 baseline."
        ),
    )
    ap.add_argument(
        "--total-sims",
        type=int, default=None,
        help=(
            "Override GumbelConfig.total_sims (default 32). Higher = more "
            "rollouts at the root. Pair with --hard-deadline-ms when "
            "raising significantly to avoid Kaggle timeouts."
        ),
    )
    ap.add_argument(
        "--hard-deadline-ms",
        type=float, default=None,
        help=(
            "Override GumbelConfig.hard_deadline_ms (default 300ms). "
            "Kaggle's actTimeout is 1000ms; we leave ~150ms headroom for "
            "the outer loop. 850ms is the practical cap."
        ),
    )
    ap.add_argument(
        "--nn-hold-neutral-prob",
        type=float,
        default=None,
        help=(
            "Mass reserved for the HOLD action in the per-planet NN "
            "prior. Default 0.05. Increase if the NN over-launches; "
            "decrease if it under-launches. Only used when "
            "--nn-checkpoint is set."
        ),
    )
    args = ap.parse_args()

    weights_override: Dict[str, float] | None = None
    if args.weights_json:
        import json
        raw = json.loads(Path(args.weights_json).read_text(encoding="utf-8"))
        unwrapped = False
        for key in ("best_weights", "best_point", "best", "weights"):
            if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                unwrapped = True
                break
        # TuRBO runner shape: {best_index, trials: [{trial, weights, ...}]}.
        # Pull the best trial's weights dict.
        if (
            not unwrapped
            and isinstance(raw, dict)
            and "best_index" in raw
            and isinstance(raw.get("trials"), list)
            and raw["best_index"] is not None
        ):
            best_i = int(raw["best_index"])
            for tr in raw["trials"]:
                if int(tr.get("trial", -1)) == best_i and isinstance(tr.get("weights"), dict):
                    raw = tr["weights"]
                    unwrapped = True
                    break
        if not isinstance(raw, dict):
            raise SystemExit(
                f"--weights-json: couldn't find a weights dict in {args.weights_json!r}"
            )
        weights_override = {k: float(v) for k, v in raw.items()}
        print(f"weights override: {len(weights_override)} keys from {args.weights_json}")

    out = Path(args.out)
    use_opp_model: bool | None = None
    if args.use_opponent_model is not None:
        use_opp_model = args.use_opponent_model == "true"
    use_macros_flag: bool | None = None
    if args.use_macros is not None:
        use_macros_flag = args.use_macros == "true"
    bundle_bot(
        args.bot, out,
        weights_override=weights_override,
        sim_move_variant=args.sim_move_variant,
        exp3_eta=args.exp3_eta,
        rollout_policy=args.rollout_policy,
        anchor_margin=args.anchor_margin,
        use_opponent_model=use_opp_model,
        nn_checkpoint=Path(args.nn_checkpoint) if args.nn_checkpoint else None,
        nn_temperature=args.nn_temperature,
        nn_hold_neutral_prob=args.nn_hold_neutral_prob,
        use_macros=use_macros_flag,
        nn_dataset_path=args.nn_dataset_path,
        total_sims=args.total_sims,
        hard_deadline_ms=args.hard_deadline_ms,
    )
    print(f"Bundled {args.bot} -> {out} ({out.stat().st_size} bytes)")
    if args.sim_move_variant:
        eta = args.exp3_eta if args.exp3_eta is not None else 0.3
        print(f"  sim_move_variant={args.sim_move_variant!r} exp3_eta={eta}")
    if args.rollout_policy:
        print(f"  rollout_policy={args.rollout_policy!r}")
    if args.anchor_margin is not None:
        print(f"  anchor_improvement_margin={args.anchor_margin}")
    if use_opp_model is not None:
        print(f"  use_opponent_model={use_opp_model}")
    if args.nn_checkpoint is not None:
        ck = Path(args.nn_checkpoint)
        sz = ck.stat().st_size if ck.exists() else 0
        print(
            f"  nn_checkpoint={args.nn_checkpoint!r} "
            f"({sz / 1024:.0f} KB raw -> ~{sz * 4 / 3 / 1024:.0f} KB base64)"
        )
        if args.nn_temperature is not None:
            print(f"  nn_temperature={args.nn_temperature}")
        if args.nn_hold_neutral_prob is not None:
            print(f"  nn_hold_neutral_prob={args.nn_hold_neutral_prob}")

    if args.smoke_test:
        _smoke_test(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
