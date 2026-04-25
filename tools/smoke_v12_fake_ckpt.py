"""End-to-end smoke for the v12 (NN-prior) bundle path using a *fake*
random-init ConvPolicy checkpoint.

Why: surfaces wiring bugs (base64 inflation, factory rewrite, kaggle
agent shape, torch import paths in the bundled file) BEFORE the real
BC checkpoint lands. The checkpoint quality doesn't matter — we just
want a single 500-turn game to complete cleanly with the NN prior
firing on every search call.

Run:
  $env:PYTHONPATH="src;."; .venv\\Scripts\\python.exe tools\\smoke_v12_fake_ckpt.py
"""
from __future__ import annotations

import sys
import time
import importlib.util
from dataclasses import asdict
from pathlib import Path

import torch  # type: ignore[import-not-found]

# Allow `from tools.bundle ...` and `from orbitwars.* ...` regardless
# of how the user launched the script.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from orbitwars.nn.conv_policy import ConvPolicy, ConvPolicyCfg


def make_fake_checkpoint(out_path: Path) -> None:
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ckpt = {"model_state": state, "cfg": asdict(cfg), "curve": {}}
    torch.save(ckpt, str(out_path))
    print(f"wrote fake BC checkpoint: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


def bundle_v12(ckpt: Path, out: Path) -> None:
    from tools.bundle import bundle_bot
    bundle_bot(
        "mcts_bot", out,
        weights_override=None,  # default heuristic weights for the smoke
        sim_move_variant="exp3",
        exp3_eta=0.3,
        nn_checkpoint=ckpt,
    )
    print(f"wrote bundle: {out}  ({out.stat().st_size / 1024:.0f} KB)")


def smoke_one_game(out: Path) -> None:
    spec = importlib.util.spec_from_file_location("v12_smoke", str(out))
    assert spec is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["v12_smoke"] = m
    t0 = time.perf_counter()
    spec.loader.exec_module(m)
    boot_dt = time.perf_counter() - t0
    print(f"bundle import + bootstrap: {boot_dt:.2f} s")
    assert callable(m.agent), "agent not callable"
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    t0 = time.perf_counter()
    env.run([m.agent, "random"])
    game_dt = time.perf_counter() - t0
    rewards = [s.reward for s in env.state]
    steps = env.state[0].observation.step
    print(f"smoke game vs random: rewards={rewards} steps={steps} wall={game_dt:.1f}s")


def main() -> int:
    tmpdir = Path("runs/v12_fake_smoke")
    tmpdir.mkdir(parents=True, exist_ok=True)
    ckpt = tmpdir / "fake_bc.pt"
    bundle_out = tmpdir / "v12_fake.py"

    make_fake_checkpoint(ckpt)
    bundle_v12(ckpt, bundle_out)
    smoke_one_game(bundle_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
