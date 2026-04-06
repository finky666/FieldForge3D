# FieldForge3D
A lightweight 3D field exploration tool with a modular plugin system.
<p align="center">
  <img src="screenshot_gyroid.png" width="600"/>
</p>
A small, self-contained desktop playground for generating **3D scalar fields** (fractal / implicit / procedural / ODE-density) and extracting an **iso-surface** for interactive viewing.

It uses a plugin system: each field is a plugin with its own parameters and a short *Formula* description displayed in the UI.

## Features

- Plugin-based 3D scalar fields (fractal, TPMS, implicit surfaces, lattice noise, attractor density, etc.)
- Interactive iso-surface preview (PyVista / VTK)
- Per-plugin parameter UI
- Formula panel (plugin-provided text)
- Safety guards against extreme polygon counts / memory blow-ups
- Reload plugins without restarting (recommended for light edits)
- Default color mode: Radius
- Default startup plugin: Gyroid (TPMS)

## Quick start

### 1) Install

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run

```bash
python main.py
```

## Project layout

- `main.py` – entry point
- `core/` – application + UI + plugin manager
- `plugins/` – field plugins
- `tools/` – optional helper scripts

## Credits

Created and maintained by **Tibor Cefan (finky666)**.  
Assisted by **ChatGPT (Majka / SuPyWomen)** for selected code sections, refactoring, and plugin formula texts.

## License

MIT – see `LICENSE`.
