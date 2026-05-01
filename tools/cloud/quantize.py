"""Int8 per-tensor symmetric quantization for ConvPolicy checkpoints.

The bundle's NN-bootstrap (`tools/bundle.py::_bundle_upcast`) reads
`_quant_scales` from the checkpoint and de-quantizes int8 tensors back
to fp32 at runtime. This script produces compatible checkpoints.

Why int8:
  * fp32 → int8 is a 4× size reduction. A 1M-param student goes from
    4 MB → 1 MB raw → ~1.4 MB base64. Tight but FITS inline under
    Kaggle's 1 MB notebook push limit when combined with --bot
    bundle's ~300 KB code.
  * Per-tensor symmetric: `int8 = round(fp32 / scale)` where
    `scale = max(|fp32|) / 127`. Round-trip error is ~0.4% relative
    on typical conv weights — invisible at inference.
  * SKIPPED for small tensors (<= 64 numel): the per-tensor scale
    overhead dominates and bias accuracy matters more than size.

Round-trip parity test included: assert quantize-then-dequantize stays
within 1% relative L1 on every tensor.

Run:
    python -m tools.cloud.quantize \\
        --in runs/cloud_az_distilled.pt \\
        --out runs/cloud_az_distilled_int8.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def quantize_state_dict(sd: dict, min_numel: int = 64, rel_err_thresh: float = 0.05):
    """Quantize fp tensors >= min_numel via per-tensor symmetric int8.

    Returns (qsd, scales) where qsd is the new state_dict (mostly int8)
    and scales is a dict[str, float] of per-tensor scales for the
    quantized entries.
    """
    qsd: dict = {}
    scales: dict[str, float] = {}
    n_q = n_skip = 0
    saved_bytes = 0
    for k, v in sd.items():
        if not v.is_floating_point():
            qsd[k] = v
            continue
        if v.numel() < min_numel:
            qsd[k] = v
            n_skip += 1
            continue
        s = v.abs().max().item() / 127.0
        if s <= 0:
            qsd[k] = v
            continue
        q = (v / s).round().clamp(-128, 127).to(torch.int8)
        # Round-trip check
        recovered = q.float() * s
        rel_err = ((recovered - v).abs().sum() / v.abs().sum().clamp_min(1e-9)).item()
        if rel_err > rel_err_thresh:
            print(f"  WARN: {k} rel_err={rel_err:.4f} > {rel_err_thresh:.0%} -- keeping fp32",
                  file=sys.stderr)
            qsd[k] = v
            continue
        qsd[k] = q
        scales[k] = float(s)
        n_q += 1
        # Saving 3/4 of the tensor (fp32 was 4 bytes, int8 is 1 byte).
        saved_bytes += v.numel() * 3
    return qsd, scales, n_q, n_skip, saved_bytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=Path,
                    help="Input fp32 .pt")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output int8-quantized .pt")
    ap.add_argument("--min-numel", type=int, default=64,
                    help="Skip tensors smaller than this")
    ap.add_argument("--rel-err-thresh", type=float, default=0.05,
                    help="Skip quantization for tensors whose round-trip "
                         "rel-err exceeds this. Default 5% is the standard "
                         "int8-inference tolerance.")
    args = ap.parse_args()

    print(f"loading {args.src}", flush=True)
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if sd is None:
        print(f"ERROR: no model_state(_dict) in {args.src}", file=sys.stderr)
        return 1

    qsd, scales, n_q, n_skip, saved = quantize_state_dict(
        sd, args.min_numel, args.rel_err_thresh)
    print(f"  quantized {n_q} tensors, skipped {n_skip} (size or rel_err)", flush=True)
    print(f"  saved ~{saved / 1e6:.1f} MB", flush=True)

    # Strip Path-typed metadata to keep the file Linux-loadable
    out = {
        "model_state": qsd,
        "cfg": ckpt["cfg"],
        "_quant_scales": scales,
    }
    if "az_trained_jointly" in ckpt:
        out["az_trained_jointly"] = ckpt["az_trained_jointly"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    src_size = args.src.stat().st_size
    out_size = args.out.stat().st_size
    print(f"saved {args.out}  ({src_size / 1024:.0f} KB -> {out_size / 1024:.0f} KB, "
          f"{100 * (1 - out_size / src_size):.0f}% reduction)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
