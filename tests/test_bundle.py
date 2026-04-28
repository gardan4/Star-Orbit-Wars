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
    """--nn-checkpoint should inline conv_policy + nn_prior + nn_value +
    obs_encode modules, base64-embed the checkpoint, and add
    `move_prior_fn=...` + `value_fn=...` to the agent factory."""
    pytest.importorskip("torch")
    ck = tmp_path / "fake_bc.pt"
    _make_fake_bc_checkpoint(ck)
    out = tmp_path / "v12_nn.py"
    bundle_bot("mcts_bot", out, nn_checkpoint=ck)
    src = out.read_text(encoding="utf-8")
    # Inlined modules (markers come from bundle.py header per-module).
    assert "# --- inlined: orbitwars/nn/conv_policy.py ---" in src
    assert "# --- inlined: orbitwars/nn/nn_prior.py ---" in src
    # nn_value carries the leaf-eval bridge (make_nn_value_fn). It MUST be
    # inlined when --nn-checkpoint is set; previously the bundler emitted
    # a runtime `from orbitwars.nn.nn_value import ...` instead, which
    # produced a Kaggle ModuleNotFoundError at boot. Regression: 2026-04-26.
    assert "# --- inlined: orbitwars/nn/nn_value.py ---" in src
    # obs_encode provides encode_grid which both nn_prior + nn_value call.
    # Without it, the prior call site hits NameError when search runs.
    assert "# --- inlined: orbitwars/features/obs_encode.py ---" in src
    # NN bootstrap + checkpoint base64.
    assert "# --- NN prior bootstrap" in src
    assert "(--nn-checkpoint" in src  # mode marker in the comment
    assert "_BUNDLE_BC_CKPT_B64 = (" in src
    assert "_bundle_move_prior_fn = make_nn_prior_fn(" in src
    assert "_bundle_value_fn = make_nn_value_fn(" in src
    # Factory rewritten to thread move_prior_fn + value_fn.
    assert "move_prior_fn=_bundle_move_prior_fn" in src
    assert "value_fn=_bundle_value_fn" in src
    assert "MCTSAgent(rng_seed=0)" not in src  # base form replaced
    # CRITICAL — bundle MUST NOT contain any unresolved `from orbitwars.*`
    # imports. Such an import would explode at Kaggle boot with
    # ModuleNotFoundError because Kaggle only ships the bundled main.py,
    # not the orbitwars package. (This is the bug that made v30, v30b,
    # v30c, v30d, v30e all error on Kaggle while v27 worked.)
    import re as _re
    leftover = _re.findall(
        r"^(?:from|import)\s+orbitwars\.[^\s]+", src, _re.MULTILINE
    )
    assert leftover == [], (
        f"bundle leaked orbitwars imports — Kaggle will fail to load "
        f"with ModuleNotFoundError. Found: {leftover}"
    )


def test_bundle_nn_checkpoint_rejects_non_mcts_bot(tmp_path):
    pytest.importorskip("torch")
    ck = tmp_path / "fake_bc.pt"
    _make_fake_bc_checkpoint(ck)
    out = tmp_path / "h.py"
    with pytest.raises(SystemExit, match="only apply to bot=mcts_bot"):
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


def test_bundle_use_macros_injects_cfg_and_inlines_module(tmp_path):
    """--use-macros should inline the macros module and emit the
    GumbelConfig setter."""
    out = tmp_path / "macros.py"
    bundle_bot("mcts_bot", out, use_macros=True)
    src = out.read_text(encoding="utf-8")
    assert "# --- inlined: orbitwars/mcts/macros.py ---" in src
    assert "_bundle_cfg.use_macros = True" in src
    assert "from orbitwars.mcts.macros import build_macro_anchors" not in src, (
        "bundled file should not contain orbitwars-package imports — they "
        "should have been stripped in favor of the inlined module"
    )


def test_bundle_use_macros_false_explicit_emits_setter(tmp_path):
    """--use-macros=false should still emit a setter (so the bundle's
    behavior is locked, not relying on dataclass defaults)."""
    out = tmp_path / "macros_off.py"
    bundle_bot("mcts_bot", out, use_macros=False)
    src = out.read_text(encoding="utf-8")
    # When use_macros=False, the modules list does NOT include macros
    # but a `_bundle_cfg.use_macros = False` setter is emitted.
    assert "_bundle_cfg.use_macros = False" in src
    assert "# --- inlined: orbitwars/mcts/macros.py ---" not in src


def test_bundle_use_macros_requires_mcts_bot(tmp_path):
    out = tmp_path / "h.py"
    with pytest.raises(SystemExit, match="use-macros only applies"):
        bundle_bot("heuristic", out, use_macros=True)


def test_bundle_macros_with_nn_and_weights_coexist(tmp_path):
    """The full v14 candidate stack: TuRBO weights + exp3 + NN prior +
    macros all in one bundle."""
    pytest.importorskip("torch")
    ck = tmp_path / "fake_bc.pt"
    _make_fake_bc_checkpoint(ck)
    out = tmp_path / "v14.py"
    bundle_bot(
        "mcts_bot", out,
        weights_override={"w_production": 20.0},
        sim_move_variant="exp3",
        exp3_eta=0.3,
        rollout_policy="fast",
        anchor_margin=0.5,
        nn_checkpoint=ck,
        use_macros=True,
    )
    src = out.read_text(encoding="utf-8")
    assert "HEURISTIC_WEIGHTS.update({" in src
    assert "_bundle_cfg.sim_move_variant = 'exp3'" in src
    assert "_bundle_cfg.rollout_policy = 'fast'" in src
    assert "_bundle_cfg.anchor_improvement_margin = 0.5" in src
    assert "_bundle_cfg.use_macros = True" in src
    assert "# --- inlined: orbitwars/mcts/macros.py ---" in src
    assert "_bundle_move_prior_fn = make_nn_prior_fn(" in src
    assert "move_prior_fn=_bundle_move_prior_fn" in src
    # End-to-end importable.
    m = _import_bundle(out, "bundle_v14_test")
    assert callable(m.agent)
    assert m._bundle_cfg.use_macros is True


def _make_int8_quantized_checkpoint(path):
    """Save a BC checkpoint with int8-quantized weights + per-tensor scales.

    Mirrors the production quantization flow: each fp32 tensor is mapped
    to int8 via per-tensor symmetric scaling. The bundle bootstrap
    dequantizes back to fp32 at load time.
    """
    torch = pytest.importorskip("torch")
    from orbitwars.nn.conv_policy import ConvPolicy, ConvPolicyCfg
    from dataclasses import asdict
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    q_state = {}
    scales = {}
    for k, v in state.items():
        if v.is_floating_point() and v.dim() > 0:
            scale = float(v.abs().max() / 127.0)
            if scale == 0:
                scale = 1.0
            q = (v / scale).round().clamp(-128, 127).to(torch.int8)
            q_state[k] = q
            scales[k] = scale
        else:
            q_state[k] = v
    ckpt = {
        "model_state": q_state, "cfg": asdict(cfg), "curve": {},
        "_quantized": "int8_per_tensor_symmetric",
        "_quant_scales": scales,
    }
    torch.save(ckpt, str(path))


def test_bundle_int8_checkpoint_dequantizes_at_load(tmp_path):
    """End-to-end: int8-quantized checkpoint should bundle, the
    bootstrap should dequantize at load, and the resulting agent should
    be callable with no precision-related errors."""
    pytest.importorskip("torch")
    ck = tmp_path / "int8.pt"
    _make_int8_quantized_checkpoint(ck)
    out = tmp_path / "v20_int8.py"
    bundle_bot("mcts_bot", out, nn_checkpoint=ck)
    src = out.read_text(encoding="utf-8")
    # Bootstrap must fetch scales and apply them.
    assert "_quant_scales" in src
    assert "v.float() * float(scales[k])" in src
    # End-to-end import + agent callable.
    m = _import_bundle(out, "bundle_int8_test")
    assert callable(m.agent)


def test_bundle_nn_dataset_path_emits_runtime_load(tmp_path):
    """--nn-dataset-path should emit a runtime torch.load(path) — no
    base64 inlining — and add ZERO bytes to the bundle beyond the
    nn_prior + conv_policy module sources."""
    pytest.importorskip("torch")
    out = tmp_path / "v17.py"
    bundle_bot(
        "mcts_bot", out,
        nn_dataset_path="/kaggle/input/orbit-wars-bc-v2/bc_warmstart_v2.pt",
    )
    src = out.read_text(encoding="utf-8")
    assert "_BUNDLE_BC_CKPT_PATH = '/kaggle/input/orbit-wars-bc-v2/" in src
    assert "_BUNDLE_BC_CKPT_B64" not in src, (
        "dataset path should not embed base64 — that's the inline-checkpoint "
        "code path"
    )
    # Both branches of the load decoder still emitted (so partial+full
    # checkpoint formats both work at runtime).
    assert "if 'model_state' in _bundle_ckpt and 'cfg' in _bundle_ckpt:" in src
    assert "elif 'model_state_dict' in _bundle_ckpt:" in src
    # Bundle stays small — no inline checkpoint.
    assert out.stat().st_size < 500_000, (
        f"bundle should be <500 KB without inline ckpt, got {out.stat().st_size}"
    )


def test_bundle_nn_dataset_and_checkpoint_mutually_exclusive(tmp_path):
    pytest.importorskip("torch")
    ck = tmp_path / "fake.pt"
    _make_fake_bc_checkpoint(ck)
    out = tmp_path / "broken.py"
    with pytest.raises(SystemExit, match="Use either --nn-checkpoint"):
        bundle_bot(
            "mcts_bot", out,
            nn_checkpoint=ck,
            nn_dataset_path="/kaggle/input/x/y.pt",
        )


def test_bundle_nn_dataset_path_requires_mcts_bot(tmp_path):
    out = tmp_path / "h.py"
    with pytest.raises(SystemExit, match="only apply to bot=mcts_bot"):
        bundle_bot("heuristic", out, nn_dataset_path="/kaggle/x/y.pt")


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


def test_bundle_no_top_level_function_name_collisions(tmp_path):
    """Regression test for PHANTOM 6.0 (2026-04-28).

    When two source modules both define a top-level function with the
    same name (e.g. ``_softmax`` in ``mcts/actions.py`` and
    ``opponent/bayes.py``), the inlined bundle has both definitions in
    the same global scope and the LATER one shadows the earlier. If
    their signatures differ, callers of the shadowed-out function get
    a TypeError at runtime — caught by the silent ``except Exception``
    in ``mcts_bot.act()``, returning ``heuristic_move``. Net effect:
    MCTS search silently degenerates to anchor-only play, yet the
    bundle imports cleanly and matches still complete.

    This test scans every recipe's bundled .py for duplicate top-level
    ``def`` names. A single duplication is a Phantom-class bug — fail
    early so it's caught by CI rather than discovered after a ladder
    submission scores at heuristic strength.
    """
    pytest.importorskip("torch")
    ck = tmp_path / "fake_bc.pt"
    _make_fake_bc_checkpoint(ck)
    # Bundle the most complex configuration so every NN module is inlined.
    out = tmp_path / "v_collision.py"
    bundle_bot(
        "mcts_bot", out,
        nn_checkpoint=ck,
        sim_move_variant="exp3",
        rollout_policy="nn_value",
        anchor_margin=0.0,
        weights_override={"score_weight": 1.0},
        use_macros=True,
    )
    src = out.read_text(encoding="utf-8")
    import collections as _co
    def_names = re.findall(r"^def\s+(\w+)", src, re.MULTILINE)
    dups = {n: c for n, c in _co.Counter(def_names).items() if c > 1}
    # ``build`` legitimately appears twice (heuristic + mcts_bot factory
    # functions); both are unused inside the bundle and don't trigger
    # runtime collisions in the agent path. Allowlist it.
    KNOWN_HARMLESS = {"build"}
    actionable = {n: c for n, c in dups.items() if n not in KNOWN_HARMLESS}
    assert actionable == {}, (
        f"bundle has top-level function name collisions — Phantom 6.0 "
        f"regression: {actionable}. Rename one of each pair in source so "
        f"the bundle inlines them with unique names."
    )
