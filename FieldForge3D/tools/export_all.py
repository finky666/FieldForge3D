#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from core.export_pipeline import export_all_pack


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch export FieldForge3D gallery / STL pack")
    parser.add_argument("--plugins-dir", default="plugins", help="Plugins directory")
    parser.add_argument("--out", default="export_pack", help="Output directory")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden plugins")
    parser.add_argument("--preview-n", type=int, default=180, help="Maximum N used for batch rendering")
    parser.add_argument("--max-size-mm", type=float, default=100.0, help="Max STL bounding box size in mm")
    args = parser.parse_args()

    manifest = export_all_pack(
        plugins_dir=Path(args.plugins_dir),
        output_dir=Path(args.out),
        include_hidden=args.include_hidden,
        preview_n=args.preview_n,
        max_size_mm=args.max_size_mm,
    )
    print(f"Done. Exported {len(manifest['records'])} records into: {args.out}")
    if manifest.get("gif"):
        print(f"Gallery GIF: {Path(args.out) / manifest['gif']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
