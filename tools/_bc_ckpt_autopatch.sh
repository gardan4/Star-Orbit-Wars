#!/bin/bash
# Watch the BC checkpoint and re-inject cfg on every modification.
# Needed because the currently-running BC was launched before my eager-save
# patched to include cfg. Workaround until BC finishes.
LAST_MTIME=0
while true; do
  if [ -f runs/bc_warmstart_small_cpu.pt ]; then
    MTIME=$(stat -c %Y runs/bc_warmstart_small_cpu.pt 2>/dev/null)
    if [ "$MTIME" != "$LAST_MTIME" ]; then
      .venv/Scripts/python.exe -c "
import torch
from dataclasses import asdict
from orbitwars.nn.conv_policy import ConvPolicyCfg
ck = 'runs/bc_warmstart_small_cpu.pt'
try:
    ckpt = torch.load(ck, map_location='cpu', weights_only=False)
    if 'cfg' not in ckpt or ckpt.get('cfg') is None:
        cfg = ConvPolicyCfg(backbone_channels=32, n_blocks=3)
        ckpt['cfg'] = asdict(cfg)
        if 'model_state_dict' in ckpt and 'model_state' not in ckpt:
            ckpt['model_state'] = ckpt['model_state_dict']
        torch.save(ckpt, ck)
        print(f'auto-patch: epoch={ckpt[\"epoch\"]}, va_acc={ckpt[\"best_val_acc\"]:.3f}')
" 2>&1 | grep auto-patch || true
      LAST_MTIME=$MTIME
    fi
  fi
  sleep 30
done
