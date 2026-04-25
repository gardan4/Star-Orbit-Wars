"""Regression tests for `tools.bundle` — the single-file packager.

The focus is the non-trivial composition steps:
  * ``--weights-json`` injection preserves HEURISTIC_WEIGHTS semantics.
  * ``--sim-move-variant`` injection emits a GumbelConfig override
    and rewrites the MCTSAgent factory to use it.
  * Bundles are importable end-to-end (parse clean, agent is callable).
  * Variant validation catches bad inputs.

These are compile-time / string-level checks plus a single live
import to confirm the bundled file is syntactically valid Python with
all dependencies resolvable.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from tools.bundle import bundle_bot


def _import_bundle(path: Path, mod_name: str):
    """Load a bundled .py file as a module and return it."""
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    assert spec is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


def test_bundle_mcts_bot_default_has_no_variant_override(tmp_path):
    """Without --sim-move-variant, the bundled file should NOT contain
    a `_bundle_cfg` injection — keeps the default-path small and
    bit-identical with the source tree's default."""
    out = tmp_path / "default.py"
    bundle_bot("mcts_bot", out)
    src = out.read_text(encoding="utf-8")
    assert "_bundle_cfg" not in src, (
        "default bundle should not inject _bundle_cfg; only variant "
        "override paths should"
    )
    # And the factory should still be the bare default form.
    assert "agent = MCTSAgent(rng_seed=0).as_kaggle_agent()" in src


def test_bundle_mcts_bot_variant_exp3_injects_config(tmp_path):
    """--sim-move-variant=exp3 should:
      * emit a `_bundle_cfg = GumbelConfig()` line
      * set sim_move_variant = 'exp3' on it
      * set exp3_eta to the provided value
      * rewrite the agent factory to pass gumbel_cfg=_bundle_cfg
    """
    out = tmp_path / "exp3.py"
    bundle_bot("mcts_bot", out, sim_move_variant="exp3", exp3_eta=0.42)
    src = out.read_text(encoding="utf-8")
    assert "_bundle_cfg = GumbelConfig()" in src
    assert "_bundle_cfg.sim_move_variant = 'exp3'" in src
    assert "_bundle_cfg.exp3_eta = 0.42" in src
    # Default factory must be rewritten to thread our cfg.
    assert (
        "agent = MCTSAgent(gumbel_cfg=_bundle_cfg, rng_seed=0).as_kaggle_agent()"
        in src
    )
    # And the bare default form must be gone (no double-inject).
    assert "agent = MCTSAgent(rng_seed=0).as_kaggle_agent()" not in src


def test_bundle_variant_exp3_default_eta_is_0_3(tmp_path):
    """Omitting --exp3-eta should default to 0.3."""
    out = tmp_path / "exp3_default_eta.py"
    bundle_bot("mcts_bot", out, sim_move_variant="exp3")
    src = out.read_text(encoding="utf-8")
    assert "_bundle_cfg.exp3_eta = 0.3" in src


def test_bundle_variant_ucb_is_explicit_noop_injection(tmp_path):
    """--sim-move-variant=ucb is a valid explicit choice (default is
    also ucb). We still emit the config so the bundle is self-documenting
    about its intent."""
    out = tmp_path / "ucb.py"
    bundle_bot("mcts_bot", out, sim_move_variant="ucb")
    src = out.read_text(encoding="utf-8")
    assert "_bundle_cfg.sim_move_variant = 'ucb'" in src


def test_bundle_variant_rejects_unknown_variant(tmp_path):
    """Unknown variant should exit (better to fail at build time than
    ship a broken bundle)."""
    out = tmp_path / "bad.py"
    with pytest.raises(SystemExit, match=r"sim-move-variant"):
        bundle_bot("mcts_bot", out, sim_move_variant="regret_matching_plus")


def test_bundle_variant_requires_mcts_bot(tmp_path):
    """--sim-move-variant shouldn't apply to non-MCTS bots (heuristic,
    noop) — they don't have a GumbelConfig to override."""
    out = tmp_path / "bad_heur.py"
    with pytest.raises(SystemExit, match=r"only applies to bot=mcts_bot"):
        bundle_bot("heuristic", out, sim_move_variant="exp3")


def test_bundle_variant_exp3_is_importable(tmp_path):
    """End-to-end: bundling with variant=exp3 produces a file that
    parses clean and exposes a callable `agent`. Catches mistakes like
    a factory-rewrite that breaks the trailing string."""
    out = tmp_path / "importable.py"
    bundle_bot("mcts_bot", out, sim_move_variant="exp3", exp3_eta=0.3)
    m = _import_bundle(out, "bundle_importable_exp3_test")
    assert callable(m.agent)
    # And the injected cfg is live on the module.
    cfg = getattr(m, "_bundle_cfg", None)
    assert cfg is not None
    assert cfg.sim_move_variant == "exp3"
    assert cfg.exp3_eta == pytest.approx(0.3)


def test_bundle_weights_plus_variant_coexist(tmp_path):
    """Weights override + variant override should both be applied and
    not interfere with each other. This is the v9 = v8-weights +
    exp3-variant composition."""
    out = tmp_path / "combined.py"
    bundle_bot(
        "mcts_bot", out,
        weights_override={"capture_likelihood_weight": 1.234},
        sim_move_variant="exp3",
        exp3_eta=0.25,
    )
    src = out.read_text(encoding="utf-8")
    # Both injections present.
    assert "HEURISTIC_WEIGHTS.update({" in src
    assert "'capture_likelihood_weight': 1.234" in src
    assert "_bundle_cfg.sim_move_variant = 'exp3'" in src
    assert "_bundle_cfg.exp3_eta = 0.25" in src
    # Weights come BEFORE the variant block (otherwise
    # the agent is constructed before weights are patched, and any
    # archetype prior built at MCTSAgent.__init__ time misses the
    # update). Check ordering explicitly.
    w_idx = src.index("HEURISTIC_WEIGHTS.update({")
    v_idx = src.index("_bundle_cfg = GumbelConfig()")
    assert w_idx < v_idx, (
        "weights must be injected before the variant cfg — otherwise "
        "MCTSAgent reads stale HEURISTIC_WEIGHTS at construction"
    )


# --- New flag coverage: --rollout-policy, --anchor-margin, --use-opponent-model ---


def test_bundle_rollout_policy_fast_injects_cfg(tmp_path):
    """--rollout-policy=fast should emit a `_bundle_cfg.rollout_policy='fast'`
    line and rewrite the agent factory to use the cfg."""
    out = tmp_path / "fast_rollout.py"
    bundle_bot("mcts_bot", out, rollout_policy="fast")
    src = out.read_text(encoding="utf-8")
    assert "_bundle_cfg = GumbelConfig()" in src
    assert "_bundle_cfg.rollout_policy = 'fast'" in src
    assert (
        "agent = MCTSAgent(gumbel_cfg=_bundle_cfg, rng_seed=0).as_kaggle_agent()"
        in src
    )


def test_bundle_rollout_policy_rejects_unknown(tmp_path):
    """Reject any rollout policy not in {heuristic, fast}."""
    out = tmp_path / "bad_rollout.py"
    with pytest.raises(SystemExit, match=r"rollout-policy"):
        bundle_bot("mcts_bot", out, rollout_policy="random")


def test_bundle_rollout_policy_requires_mcts_bot(tmp_path):
    """--rollout-policy shouldn't apply to non-MCTS bots."""
    out = tmp_path / "heur_rollout.py"
    with pytest.raises(SystemExit, match=r"only applies to bot=mcts_bot"):
        bundle_bot("heuristic", out, rollout_policy="fast")


def test_bundle_anchor_margin_injects_cfg(tmp_path):
    """--anchor-margin=0.5 should emit `_bundle_cfg.anchor_improvement_margin=0.5`."""
    out = tmp_path / "margin.py"
    bundle_bot("mcts_bot", out, anchor_margin=0.5)
    src = out.read_text(encoding="utf-8")
    assert "_bundle_cfg.anchor_improvement_margin = 0.5" in src


def test_bundle_anchor_margin_requires_mcts_bot(tmp_path):
    """--anchor-margin shouldn't apply to non-MCTS bots."""
    out = tmp_path / "heur_margin.py"
    with pytest.raises(SystemExit, match=r"only applies to bot=mcts_bot"):
        bundle_bot("heuristic", out, anchor_margin=0.5)


def test_bundle_use_opp_model_false_threads_factory_kwarg(tmp_path):
    """--use-opponent-model=false should add `use_opponent_model=False` to
    the MCTSAgent factory (NOT to GumbelConfig — it's an MCTSAgent arg)."""
    out = tmp_path / "no_opp_model.py"
    bundle_bot("mcts_bot", out, use_opponent_model=False)
    src = out.read_text(encoding="utf-8")
    # Factory should thread the kwarg.
    assert "use_opponent_model=False" in src
    # The original bare factory must be gone.
    assert "agent = MCTSAgent(rng_seed=0).as_kaggle_agent()" not in src


def test_bundle_use_opp_model_alone_skips_bundle_cfg(tmp_path):
    """If ONLY use_opponent_model is set (no cfg attrs), we shouldn't
    emit a _bundle_cfg block — it would be dead code. But the factory
    should still get rewritten to thread the kwarg."""
    out = tmp_path / "only_opp.py"
    bundle_bot("mcts_bot", out, use_opponent_model=False)
    src = out.read_text(encoding="utf-8")
    assert "_bundle_cfg = GumbelConfig()" not in src
    assert "MCTSAgent(rng_seed=0, use_opponent_model=False)" in src


def test_bundle_all_mcts_overrides_combined_importable(tmp_path):
    """End-to-end: rollout_policy=fast + anchor_margin=0.5 +
    use_opponent_model=False all combined should produce a clean,
    importable bundle. This is the v10 experimental config."""
    out = tmp_path / "v10_experimental.py"
    bundle_bot(
        "mcts_bot", out,
        weights_override={"w_production": 7.5},
        rollout_policy="fast",
        anchor_margin=0.5,
        use_opponent_model=False,
    )
    src = out.read_text(encoding="utf-8")
    assert "HEURISTIC_WEIGHTS.update({" in src
    assert "_bundle_cfg.rollout_policy = 'fast'" in src
    assert "_bundle_cfg.anchor_improvement_margin = 0.5" in src
    assert "use_opponent_model=False" in src
    assert "gumbel_cfg=_bundle_cfg" in src
    # Importability check.
    m = _import_bundle(out, "bundle_v10_test")
    assert callable(m.agent)
    cfg = m._bundle_cfg
    assert cfg.rollout_policy == "fast"
    assert cfg.anchor_improvement_margin == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# --nn-checkpoint tests (W4-5 NN prior bundle path)
# ---------------------------------------------------------------------------


def _make_fake_bc_checkpoint(path: Path, partial: bool = False) -> None:
    """Save a tiny default-cfg ConvPolicy as a BC-style .pt file.

    Used by --nn-checkpoint tests so they don't depend on a real BC run.
    """
    torch = pytest.importorskip("torch")
    from orbitwars.nn.conv_policy import ConvPolicy, ConvPolicyCfg
    from dataclasses import asdict
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if partial:
        ckpt = {
            "model_state_dict": state,
            "best_val_acc": 0.42,
            "epoch": 1,
            "_partial": True,
        }
    else:
        ckpt = {"model_state": state, "cfg": asdict(cfg), "curve": {}}
    torch.save(ckpt, str(path))


def test_bundle_nn_checkpoint_inlines_modules_and_factory(tmp_path):
    """--nn-checkpoint should inline conv_policy + nn_prior modules,
    base64-embed the checkpoint, and add `move_prior_fn=...` to the
    agent factory."""
    pytest.importorskip("torch")
    ck = tmp_path / "fake_bc.pt"
    _make_fake_bc_checkpoint(ck)
    out = tmp_path / "v12_nn.py"
    bundle_bot("mcts_bot", out, nn_checkpoint=ck)
    src = out.read_text(encoding="utf-8")
    # Inlined modules (markers come from bundle.py header per-module).
    assert "# --- inlined: orbitwars/nn/conv_policy.py ---" in src
    assert "# --- inlined: orbitwars/nn/nn_prior.py ---" in src
    # NN bootstrap + checkpoint base64.
    assert "# --- NN prior bootstrap (--nn-checkpoint) ---" in src
    assert "_BUNDLE_BC_CKPT_B64 = (" in src
    assert "_bundle_move_prior_fn = make_nn_prior_fn(" in src
    # Factory rewritten to thread move_prior_fn.
    assert "move_prior_fn=_bundle_move_prior_fn" in src
    assert "MCTSAgent(rng_seed=0)" not in src  # base form replaced


def test_bundle_nn_checkpoint_rejects_non_mcts_bot(tmp_path):
    pytest.importorskip("torch")
    ck = tmp_path / "fake_bc.pt"
    _make_fake_bc_checkpoint(ck)
    out = tmp_path / "h.py"
    with pytest.raises(SystemExit, match="nn-checkpoint only applies"):
        bundle_bot("heuristic", out, nn_checkpoint=ck)


def test_bundle_nn_checkpoint_missing_file_raises(tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "v12.py"
    with pytest.raises(SystemExit, match="file not found"):
        bundle_bot("mcts_bot", out, nn_checkpoint=tmp_path / "nope.pt")


def test_bundle_nn_checkpoint_partial_format_inlines_cleanly(tmp_path):
    """The eager-save partial format (model_state_dict only) should bundle
    just as cleanly as the full format. The bootstrap branches at runtime."""
    pytest.importorskip("torch")
    ck = tmp_path / "partial.pt"
    _make_fake_bc_checkpoint(ck, partial=True)
    out = tmp_path / "v12_partial.py"
    bundle_bot("mcts_bot", out, nn_checkpoint=ck)
    src = out.read_text(encoding="utf-8")
    # Both branches must be emitted so the runtime decode can pick.
    assert "if 'model_state' in _bundle_ckpt and 'cfg' in _bundle_ckpt:" in src
    assert "elif 'model_state_dict' in _bundle_ckpt:" in src


def test_bundle_nn_checkpoint_temperature_propagates(tmp_path):
    pytest.importorskip("torch")
    ck = tmp_path / "fake_bc.pt"
    _make_fake_bc_checkpoint(ck)
    out = tmp_path / "v12_temp.py"
    bundle_bot(
        "mcts_bot", out, nn_checkpoint=ck,
        nn_temperature=0.5, nn_hold_neutral_prob=0.2,
    )
    src = out.read_text(encoding="utf-8")
    # Both the temperature and hold_neutral_prob are emitted in the
    # make_nn_prior_fn(...) call inside the bootstrap.
    assert "temperature=0.5" in src
    assert "hold_neutral_prob=0.2" in src


def test_bundle_nn_checkpoint_with_weights_and_variant_coexist(tmp_path):
    """The NN bundle is the v12 candidate: nn_checkpoint + tuned weights
    (TuRBO-v3) + exp3 variant should all compose cleanly."""
    pytest.importorskip("torch")
    ck = tmp_path / "fake_bc.pt"
    _make_fake_bc_checkpoint(ck)
    out = tmp_path / "v12_full.py"
    bundle_bot(
        "mcts_bot", out,
        weights_override={"w_production": 20.0, "mult_enemy": 5.0},
        sim_move_variant="exp3",
        exp3_eta=0.3,
        nn_checkpoint=ck,
    )
    src = out.read_text(encoding="utf-8")
    assert "HEURISTIC_WEIGHTS.update({" in src
    assert "_bundle_cfg.sim_move_variant = 'exp3'" in src
    assert "_bundle_cfg.exp3_eta = 0.3" in src
    assert "_bundle_move_prior_fn = make_nn_prior_fn(" in src
    # Factory has BOTH gumbel_cfg AND move_prior_fn kwargs.
    assert "gumbel_cfg=_bundle_cfg" in src
    assert "move_prior_fn=_bundle_move_prior_fn" in src


def test_bundle_nn_checkpoint_importable_and_agent_callable(tmp_path):
    """End-to-end: bundle a fake-checkpoint v12, import it, confirm
    the agent is callable. This catches torch import issues, base64
    decode issues, and ConvPolicy state_dict load mismatches."""
    pytest.importorskip("torch")
    ck = tmp_path / "fake_bc.pt"
    _make_fake_bc_checkpoint(ck)
    out = tmp_path / "v12_callable.py"
    bundle_bot("mcts_bot", out, nn_checkpoint=ck)
    m = _import_bundle(out, "bundle_v12_test")
    assert callable(m.agent)
    # Verify the bootstrap actually built a model, not just emitted text.
    assert hasattr(m, "_bundle_model")
    assert hasattr(m, "_bundle_move_prior_fn")
