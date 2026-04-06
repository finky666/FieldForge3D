from __future__ import annotations
from pathlib import Path
import re
import sys

# Detect Slovak/Czech leftovers in source text.
# - diacritics
# - common SK/CZ words
DIACRITICS_RE = re.compile(r"[áäčďéíľĺňóôŕšťúýžÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ]")
BAD_WORDS = [
    "mapovanie","vyhladen","hustot","poznam","kock","zlat","jadro","ramen","hrbka",
    "vstup","aby sa","nezmenila","zvysuje","zhustenie","mierky","skladme",
]

ROOTS = [Path("core"), Path("plugins"), Path("README.md")]

def scan_file(p: Path):
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    hits=[]
    for i,line in enumerate(txt.splitlines(), start=1):
        low=line.lower()
        if DIACRITICS_RE.search(line):
            hits.append((p.as_posix(), i, "diacritics", line.strip()))
            continue
        for w in BAD_WORDS:
            if w in low:
                hits.append((p.as_posix(), i, w, line.strip()))
                break
    return hits

def main():
    all_hits=[]
    for r in ROOTS:
        if r.is_dir():
            for p in r.rglob("*.py"):
                all_hits.extend(scan_file(p))
        elif r.exists():
            all_hits.extend(scan_file(r))
    if not all_hits:
        print("[OK] No obvious SK/CZ leftovers found.")
        return 0
    print("[FAIL] Found potential SK/CZ leftovers:")
    for fp,ln,tag,line in all_hits:
        print(f" - {fp}:{ln}  [{tag}]  {line}")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
