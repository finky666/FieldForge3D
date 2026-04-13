# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

from __future__ import annotations

import re
import shutil
from pathlib import Path
from datetime import datetime

# Text-only safe edit: we do NOT import plugins.
# We add missing PLUGIN_META["formula"] key (preferred), not legacy FORMULA variable.

RE_PLUGIN_META_BLOCK = re.compile(r'^(PLUGIN_META\s*=\s*\{.*?\}\s*)$', re.S | re.M)
RE_ID = re.compile(r'["\']id["\']\s*:\s*["\']([^"\']+)["\']')
RE_NAME = re.compile(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']')
RE_HAS_META_FORMULA = re.compile(r'["\']formula["\']\s*:')


def build_formula_value(pid: str, pname: str, filename: str) -> str:
    # Keep it short; user fills it later.
    # Use explicit \n to keep the dict readable.
    return (
        f"{pname}\\n"
        f"Plugin: {pid}  ({filename})\\n\\n"
        "Formulas:\\n"
        "  - TODO\\n"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    plugins_dir = root / "plugins"
    bak_dir = plugins_dir / "_bak_formulas"
    bak_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    changed = 0
    skipped = 0
    no_meta = 0

    for path in sorted(plugins_dir.glob("*.py")):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue

        txt = path.read_text(encoding="utf-8", errors="ignore")

        m = RE_PLUGIN_META_BLOCK.search(txt)
        if not m:
            no_meta += 1
            continue

        meta_block = m.group(1)
        if RE_HAS_META_FORMULA.search(meta_block):
            skipped += 1
            continue

        mid = RE_ID.search(meta_block)
        if not mid:
            no_meta += 1
            continue

        pid = mid.group(1).strip()
        mname = RE_NAME.search(meta_block)
        pname = mname.group(1).strip() if mname else pid

        # Insert "formula" key just before the closing '}' of PLUGIN_META.
        # We keep indentation consistent with existing dict.
        formula_value = build_formula_value(pid, pname, path.name)

        # Find last '}' in the meta block (safe enough because meta block is minimal)
        idx_close = meta_block.rfind("}")
        if idx_close < 0:
            no_meta += 1
            continue

        # Determine indentation (use 4 spaces by convention)
        indent = "    "

        insertion = (
            f"{indent}\"formula\": (\n"
            f"{indent}{indent}{formula_value!r}\n"
            f"{indent}),\n"
        )

        new_meta_block = meta_block[:idx_close] + insertion + meta_block[idx_close:]

        # backup
        bak_path = bak_dir / f"{path.stem}.{ts}.bak.py"
        shutil.copy2(path, bak_path)

        new_txt = txt[:m.start(1)] + new_meta_block + txt[m.end(1):]
        path.write_text(new_txt, encoding="utf-8")
        changed += 1
        print(f"[OK] {path.name}: added PLUGIN_META['formula'] (backup -> {bak_path.name})")

    print("\n--- SUMMARY ---")
    print(f"Changed : {changed}")
    print(f"Skipped : {skipped} (already had meta formula)")
    print(f"No META : {no_meta} (missing PLUGIN_META or id)")
    print(f"Backups : {bak_dir}")


if __name__ == "__main__":
    main()
