# FieldForge3D 1.0 – Export Edition

## Major changes

- Added **File** menu export workflow
  - Export UI screenshot
  - Export clean render
  - Export current STL
  - Export all pack
- Added **batch export pipeline** in `core/export_pipeline.py`
- Added CLI batch exporter: `tools/export_all.py`
- Added printable STL scaling to **max 100 mm**
- Added new plugins:
  - `organic_blob`
  - `ripple_shell`
  - hidden `dancing_eggs`
- Added hidden Easter egg unlock through **Help → About**
- Updated README for Windows + Linux + Ubuntu 24.04 / xcb notes
- Promoted project wording to **1.0**

## Export all pack output

```text
fieldforge_export_YYYYMMDD_HHMMSS/
  current_ui.png
  current_clean.png
  images/
  print/
  gallery.gif
  manifest.json
```

## Author / credits

Project owner: **Tibor Čefan (finky666)**

1.0 refactor / export workflow / new plugin integration:
**ChatGPT (Majka / SuPyWomen)**
