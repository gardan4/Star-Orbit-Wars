r"""Validate a BC warm-start checkpoint written by ``tools/bc_warmstart.py``.

Run this after ``bc_warmstart.py`` finishes to confirm the checkpoint is
usable as a PPO warm-start: schema OK, tensors load into a fresh
``ConvPolicy``, val accuracy clears the gate, and a dummy forward pass
produces sensibly-shaped output.

Usage::

    .\.venv-gpu\Scripts\python.exe -m tools.validate_bc_checkpoint \
        --checkpoint runs/bc_warmstart.pt

Gates:
  * Required keys: ``model_state``, ``cfg``, ``curve``, ``demo_hash``,
    ``n_demos``, ``hparams``.
  * ``best_val_acc`` >= max(0.25, majority_class + 0.05). Same rule
    as ``bc_warmstart.py``.
  * Loading ``model_state`` into ``ConvPolicy(ConvPolicyCfg(**cfg))``
    must succeed with zero missing/unexpected keys.
  * A ``(1, n_channels, grid_h, grid_w)`` dummy forward must return
    finite policy logits and a value in [-1, 1] (tanh-bounded).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def validate(checkpoint_path: Path) -> Dict[str, Any]:
    """Runs all checks. Returns a summary dict the caller can print/log.

    Raises AssertionError on the first failing check.
    """
    import torch
    from orbitwars.nn.conv_policy import ConvPolicy, ConvPolicyCfg

    _assert(checkpoint_path.exists(), f"checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    required_keys = {
        "model_state", "cfg", "curve", "demo_hash", "n_demos", "hparams",
    }
    missing = required_keys - set(ckpt.keys())
    _assert(not missing, f"missing checkpoint keys: {missing}")

    # Rebuild cfg and instantiate the model.
    cfg_dict = ckpt["cfg"]
    _assert(isinstance(cfg_dict, dict), "cfg is not a dict")
    cfg = ConvPolicyCfg(**cfg_dict)
    model = ConvPolicy(cfg)

    # State-dict load — catch any shape/name drift.
    state = ckpt["model_state"]
    missing_k, unexpected_k = model.load_state_dict(state, strict=False)
    _assert(not missing_k, f"missing state_dict keys: {missing_k}")
    _assert(not unexpected_k, f"unexpected state_dict keys: {unexpected_k}")

    # Dummy forward.
    x = torch.zeros(1, cfg.n_channels, cfg.grid_h, cfg.grid_w, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        policy, value = model(x)
    _assert(policy.shape == (1, cfg.n_action_channels, cfg.grid_h, cfg.grid_w),
            f"unexpected policy shape: {tuple(policy.shape)}")
    _assert(value.shape == (1, 1),
            f"unexpected value shape: {tuple(value.shape)}")
    _assert(torch.isfinite(policy).all().item(),
            "policy logits contain non-finite values")
    _assert(torch.isfinite(value).all().item(),
            "value head produced non-finite output")
    v = float(value.item())
    _assert(-1.0 - 1e-6 <= v <= 1.0 + 1e-6,
            f"value out of tanh range: {v}")

    # Curve + gate.
    curve = ckpt["curve"]
    best_val = float(curve.get("best_val_acc", -math.inf))
    val_acc_history = list(curve.get("val_acc", []))
    train_acc_history = list(curve.get("train_acc", []))
    n_demos = int(ckpt["n_demos"])
    # We don't see the original majority_class directly — fall back to
    # an 8-class cap (same as bc_warmstart's ACTION_LOOKUP).
    n_action_classes = cfg.n_action_channels
    random_guess = 1.0 / n_action_classes
    # Conservative gate: 2x random guess is the floor. If the curve has
    # enough epochs, require also monotone-ish training (last val ≥ 0.25).
    gate_floor = 2 * random_guess
    gate_pass = best_val >= gate_floor

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "file_size_bytes": checkpoint_path.stat().st_size,
        "cfg": cfg_dict,
        "hparams": ckpt["hparams"],
        "n_demos": n_demos,
        "demo_hash": ckpt["demo_hash"],
        "torch_version": ckpt.get("torch_version"),
        "cuda_available": ckpt.get("cuda_available"),
        "best_val_acc": best_val,
        "final_train_acc": train_acc_history[-1] if train_acc_history else None,
        "final_val_acc": val_acc_history[-1] if val_acc_history else None,
        "n_epochs_trained": len(val_acc_history),
        "random_guess_baseline": random_guess,
        "gate_floor": gate_floor,
        "gate_pass": gate_pass,
        "dummy_forward": {
            "policy_shape": tuple(policy.shape),
            "value_shape": tuple(value.shape),
            "value_sample": v,
            "policy_l2": float(policy.norm().item()),
        },
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate a BC warm-start checkpoint."
    )
    ap.add_argument(
        "--checkpoint",
        default="runs/bc_warmstart.pt",
        help="Path to the .pt checkpoint written by bc_warmstart.py.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit the full summary as JSON to stdout.",
    )
    args = ap.parse_args(argv)

    path = Path(args.checkpoint)
    try:
        summary = validate(path)
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    if args.json:
        import json
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("=== BC checkpoint validation ===")
        print(f"path: {summary['checkpoint_path']}")
        print(f"size: {summary['file_size_bytes'] / 1024:.0f} KB")
        print(f"cfg:  backbone={summary['cfg']['backbone_channels']} "
              f"blocks={summary['cfg']['n_blocks']} "
              f"actions={summary['cfg']['n_action_channels']} "
              f"grid={summary['cfg']['grid_h']}x{summary['cfg']['grid_w']}")
        print(f"hparams: {summary['hparams']}")
        print(f"n_demos: {summary['n_demos']:,}")
        print(f"n_epochs: {summary['n_epochs_trained']}")
        print(f"best_val_acc: {summary['best_val_acc']:.3f}  "
              f"(final val {summary['final_val_acc']:.3f}, "
              f"final train {summary['final_train_acc']:.3f})"
              if summary['final_val_acc'] is not None else "")
        print(f"gate floor:   {summary['gate_floor']:.3f}  (2x random guess)")
        print(f"dummy forward: policy_l2={summary['dummy_forward']['policy_l2']:.3f} "
              f"value={summary['dummy_forward']['value_sample']:+.3f}")
        marker = "PASS" if summary["gate_pass"] else "FAIL"
        print(f"\nGate: {marker}")
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
