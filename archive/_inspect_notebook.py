"""Quick notebook cell dump."""
import json
import sys
from pathlib import Path

nb_path = Path(sys.argv[1])
nb = json.loads(nb_path.read_text(encoding="utf-8"))
cells = nb["cells"]
print(f"total cells: {len(cells)}")
for i, c in enumerate(cells):
    src = c["source"]
    if isinstance(src, list):
        src = "".join(src)
    first = src[:120].replace("\n", " | ")
    print(f"cell {i}: type={c['cell_type']} len={len(src)} first120={first!r}")
