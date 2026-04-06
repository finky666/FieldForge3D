# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# core/plugins.py
from __future__ import annotations

import os
import sys
import traceback
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PluginInfo:
    id: str
    name: str
    module: Any
    path: Path


class PluginManager:
    """Plugin manager (English)."""

    def __init__(self, plugins_dir: Path):
        self.plugins_dir = Path(plugins_dir).resolve()
        self.plugins: dict[str, PluginInfo] = {}
        self._load_errors: list[str] = []

    @property
    def load_errors(self) -> list[str]:
        return list(self._load_errors)

    def scan(self) -> None:
        self.plugins.clear()
        self._load_errors.clear()

        self.plugins_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(self.plugins_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                mod = self._load_module_from_path(path)
                meta = getattr(mod, "PLUGIN_META", None)
                if not isinstance(meta, dict):
                    raise ValueError("PLUGIN_META must be a dict")

                pid = str(meta.get("id", "")).strip()
                pname = str(meta.get("name", "")).strip()
                if not pid or not pname:
                    raise ValueError("PLUGIN_META must have 'id' and 'name'")

                for fn in ("get_defaults", "build_ui", "compute"):
                    if not hasattr(mod, fn):
                        raise ValueError(f"Plugin is missing function {fn}()")

                self.plugins[pid] = PluginInfo(id=pid, name=pname, module=mod, path=path)

            except Exception as e:
                tb = traceback.format_exc()
                self._load_errors.append(f"{path.name}: {e}\n{tb}")

    def get(self, plugin_id: str) -> PluginInfo | None:
        return self.plugins.get(plugin_id)

    def list(self) -> list[PluginInfo]:
        return list(self.plugins.values())

    def _load_module_from_path(self, path: Path):
        name = f"wb_plugin_{path.stem}_{abs(hash(str(path)))}"

        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create import spec for {path}")

        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
