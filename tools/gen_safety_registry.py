# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

from __future__ import annotations
import re
from pathlib import Path

RE_ID = re.compile(r'PLUGIN_META\s*=\s*\{[^}]*["\']id["\']\s*:\s*["\']([^"\']+)["\']', re.S)

def main():
    root = Path(__file__).resolve().parents[1]
    plugins_dir = root / "plugins"
    out_path = root / "core" / "safety_registry.py"

    ids = []
    for p in sorted(plugins_dir.glob("*.py")):
        if p.name.startswith("_") or p.name == "__init__.py":
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        m = RE_ID.search(txt)
        if m:
            ids.append(m.group(1).strip())

    lines = []
    lines.append("# core/safety_registry.py")
    lines.append("# Auto-generated skeleton. Adjust per plugin if needed.")
    lines.append("")
    lines.append("GLOBAL_LIMITS = {")
    lines.append("    'MAX_N': 520,")
    lines.append("    'MAX_POLYS': 4_000_000,")
    lines.append("    'HARD_STOP': 18_000_000,")
    lines.append("}")
    lines.append("")
    lines.append("PLUGIN_LIMITS = {")
    for pid in ids:
        lines.append(f"    # '{pid}': {{'MAX_N': 240, 'MAX_POLYS': 2_500_000}},")
    lines.append("}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote {out_path} (edit overrides manually).")

if __name__ == "__main__":
    main()
