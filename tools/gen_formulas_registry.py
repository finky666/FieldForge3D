from __future__ import annotations
import re
from pathlib import Path

RE_ID = re.compile(
    r'PLUGIN_META\s*=\s*\{[^}]*["\']id["\']\s*:\s*["\']([^"\']+)["\']',
    re.S
)
RE_NAME = re.compile(
    r'PLUGIN_META\s*=\s*\{[^}]*["\']name["\']\s*:\s*["\']([^"\']+)["\']',
    re.S
)
RE_FORMULA = re.compile(
    r'FORMULA\s*=\s*r?([\'"]{3})(.*?)\1',
    re.S
)

def main():
    root = Path(__file__).resolve().parents[1]
    plugins_dir = root / "plugins"
    out_path = root / "core" / "formulas_registry.py"

    items = []
    for p in sorted(plugins_dir.glob("*.py")):
        if p.name.startswith("_") or p.name == "__init__.py":
            continue

        txt = p.read_text(encoding="utf-8", errors="ignore")

        m_id = RE_ID.search(txt)
        if not m_id:
            continue
        pid = m_id.group(1).strip()

        m_name = RE_NAME.search(txt)
        pname = m_name.group(1).strip() if m_name else pid

        m_formula = RE_FORMULA.search(txt)
        formula = m_formula.group(2).strip() if m_formula else None

        items.append((pid, pname, p.name, formula))

    lines = []
    lines.append("# core/formulas_registry.py")
    lines.append("# Auto-generated from plugin FORMULA strings.")
    lines.append("# You may still edit entries manually if you want custom overrides.")
    lines.append("")
    lines.append("FORMULAS = {")

    for pid, pname, fname, formula in items:
        lines.append(f'    "{pid}": r"""\\')
        if formula:
            lines.append(formula)
        else:
            lines.append(f"{pname}  ({fname})")
            lines.append("")
            lines.append("Formulas:")
            lines.append("  - TODO")
        lines.append('""",')
        lines.append("")

    lines.append("}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote {out_path} with {len(items)} entries.")

if __name__ == "__main__":
    main()