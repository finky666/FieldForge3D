# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# core/presets.py
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _preset_file_path() -> Path:
    """Return the path where presets should be stored.

    - In dev (python main.py): store next to project root (next to main.py).
    - In frozen/EXE: store next to the executable.
    """
    try:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent / "user_presets.json"
    except Exception:
        pass
    # project root: .../FieldForge3D/core -> .../FieldForge3D
    return Path(__file__).resolve().parents[1] / "user_presets.json"


def load_presets() -> Dict[str, Any]:
    p = _preset_file_path()
    if not p.exists():
        return {"version": 1, "plugins": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "plugins": {}}
    if not isinstance(data, dict):
        return {"version": 1, "plugins": {}}
    data.setdefault("version", 1)
    data.setdefault("plugins", {})
    if not isinstance(data["plugins"], dict):
        data["plugins"] = {}
    return data


def save_presets(data: Dict[str, Any]) -> None:
    p = _preset_file_path()
    try:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # ignore; app should not crash because of IO
        pass


def get_plugin_preset(plugin_id: str) -> Optional[Dict[str, Any]]:
    data = load_presets()
    preset = data.get("plugins", {}).get(str(plugin_id), None)
    return preset if isinstance(preset, dict) else None


def set_plugin_preset(plugin_id: str, preset: Dict[str, Any]) -> None:
    data = load_presets()
    data.setdefault("plugins", {})
    data["plugins"][str(plugin_id)] = dict(preset)
    save_presets(data)


def clear_plugin_preset(plugin_id: str) -> None:
    data = load_presets()
    plugins = data.get("plugins", {})
    if isinstance(plugins, dict) and str(plugin_id) in plugins:
        del plugins[str(plugin_id)]
        save_presets(data)
