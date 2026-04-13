# FieldForge 3D (FieldForge3D)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# core/app.py
from __future__ import annotations

import os
import time
import math
import numpy as np
from pathlib import Path

import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt6 import QtWidgets, QtCore, QtGui

from core.panel import PluginHostPanel
from core.plugins import PluginManager
from core.worker import ComputeWorker
from core.memory_guard import guard as mem_guard, suggest_lower_N

from core.formulas_registry import FORMULAS
from core.safety_registry import GLOBAL_LIMITS, PLUGIN_LIMITS, clamp_params
from core.presets import get_plugin_preset, set_plugin_preset, clear_plugin_preset
from core.export_pipeline import export_all_pack, export_stl


APP_VERSION = "1.1.0"

DEFAULT_STATE = dict(
    DARK_BG=True,
    SHOW_AXES=False,
    SHOW_BOUNDS=False,
    SMOOTH_SHADING=True,
    ANTIALIAS=True,

    COLOR_MODE="radius",       # value | height | radius
    COLORMAP_DARK="turbo",
    COLORMAP_LIGHT="viridis",
    SHOW_SCALAR_BAR=False,
)


class FieldWorkbenchApp(QtCore.QObject):
    def __init__(self):
        super().__init__()

        # --- plotter & window ---
        self.plotter = BackgroundPlotter(title="FieldForge3D", auto_update=True)
        self.win: QtWidgets.QMainWindow = self.plotter.app_window


        # --- status bar (full-width, fixed height, single-line) ---
        self.status = QtWidgets.QStatusBar(self.win)
        self.status.setSizeGripEnabled(False)
        self.status.setStyleSheet("QStatusBar{color:#dddddd;}")
        self.win.setStatusBar(self.status)

        self._status_lock = False
        self._idle_status_text = "Ready."

        self._status_label = QtWidgets.QLabel(self._idle_status_text, self.win)
        self._status_label.setStyleSheet("QLabel{padding:0px 6px;}")
        self._status_label.setMinimumHeight(18)
        self._status_label.setWordWrap(False)

        # permanent widget stretches across the statusbar
        self.status.addPermanentWidget(self._status_label, 1)



        # --- state ---
        self.state = dict(DEFAULT_STATE)
        self.dark_bg = bool(self.state["DARK_BG"])
        self.axes_visible = bool(self.state["SHOW_AXES"])
        self._first_render_done = False

        # caches
        self._last_surf: pv.PolyData | None = None

        # worker thread
        self._thread: QtCore.QThread | None = None
        self._worker: ComputeWorker | None = None
        # --- pending recompute (spam-click / auto-cycle safety) ---
        self._pending_recompute = False
        self._pending_reason = ""


        # --- plugin manager (relative path!) ---
        root = Path(__file__).resolve().parent.parent  # .../3d_field_workbench
        self.plugins_dir = (root / "plugins").resolve()
        self.pm = PluginManager(self.plugins_dir)
        self.pm.scan()

        # --- safety limits (global + per-plugin overrides) ---
        self._global_limits = dict(GLOBAL_LIMITS)

        # --- left panel ---
        self.panel = PluginHostPanel()
        dock = QtWidgets.QDockWidget("Controls", self.win)
                # Wrap left panel in scroll area to avoid overflow overlapping status bar
        self._panel_scroll = QtWidgets.QScrollArea()
        self._panel_scroll.setWidgetResizable(True)
        self._panel_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._panel_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._panel_scroll.setWidget(self.panel)
        dock.setWidget(self._panel_scroll)
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.win.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        dock.setFloating(False)
        dock.setMinimumWidth(380)
        self._dock = dock

        # --- formulas dock (floating / dockable) ---
        self._formulas_dock = QtWidgets.QDockWidget("Formulas", self.win)
        self._formulas_dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._formulas_text = QtWidgets.QPlainTextEdit(self.win)
        self._formulas_text.setReadOnly(True)
        self._formulas_text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._formulas_text.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        self._formulas_text.setPlainText("Formulas: (select a plugin)")
        self._formulas_text.setStyleSheet("QPlainTextEdit { padding: 6px; }")
        self._formulas_dock.setWidget(self._formulas_text)
        self._formulas_dock.setMinimumWidth(420)
        self.win.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._formulas_dock)
        self._formulas_dock.setVisible(False)

        self.panel.recompute_clicked.connect(self.recompute)
        self.panel.reload_plugins_clicked.connect(self.reload_plugins)
        self.panel.plugin_changed.connect(self._on_plugin_changed)
        self.panel.save_preset_clicked.connect(self._on_save_preset)
        self.panel.clear_preset_clicked.connect(self._on_clear_preset)

        # init plugins into UI
        self._populate_plugins(initial=True)


        # safe picking patch
        self._patch_safe_picking()

        # menu
        self._build_menu()

        # keys (PyVista)
        self.plotter.add_key_event("a", self.toggle_axes)
        self.plotter.add_key_event("b", self.toggle_bg)
        self.plotter.add_key_event("s", self.screenshot)
        self.plotter.add_key_event("r", self.reset_view)

        # apply view options
        self._apply_view_options()

        # --- autopilot / fullscreen / cycling ---
        self._is_fullscreen = False

        self._autopilot_on = False
        self._autopilot_t = 0.0
        self._autopilot_speed_deg = 18.0
        self._autopilot_breathe = 0.06
        self._autopilot_phase = 0.0

        self._cycle_on = False
        self._cycle_seconds = 18.0
        self._cycle_next_t = 0.0

        # Auto-cycle safety defaults
        self._cycle_max_n = 280
        self._cycle_blacklist = {
            "mandelbox_like",
            "mandelbox",
            "menger_sponge",
        }

        self._auto_timer = QtCore.QTimer(self.win)
        self._auto_timer.setInterval(33)  # ~30 FPS
        self._auto_timer.timeout.connect(self._on_auto_tick)
        self._auto_timer.start()

        self._auto_elapsed = QtCore.QElapsedTimer()
        self._auto_elapsed.start()

        # Memory Guard policy:
        # - App must always start smoothly (no guard popup on startup).
        # - Guard should warn ONLY on truly bad settings ("red").
        # - Once user chooses Continue for a given (plugin_id, N), don't nag again in this session.
        self._mem_guard_ack = set()     # {(plugin_id, N)} accepted this session
        self._suppress_guard_once = True
        # NOTE: do not auto-recompute on startup (keeps startup silent)
        # self.recompute()

    # =========================
    # Plugin handling
    # =========================
    def _populate_plugins(self, initial: bool = False):
        items = []
        for p in self.pm.list():
            try:
                meta = getattr(p.module, "PLUGIN_META", {}) or {}
            except Exception:
                meta = {}

            if bool(meta.get("hidden", False)):
                continue

            items.append((p.id, p.name))

        if not items:
            self.set_status("No plugins found in plugins/.", 0, "err", sticky=True)
            try:
                self.panel.set_plugins([])
            except Exception:
                pass
            return

        prefer = ["gyroid", "wave_lattice", "organic_blob", "torus", "superquadric", "metaballs"]
        ids = [pid for pid, _ in items]
        select_id = items[0][0]
        for p in prefer:
            if p in ids:
                select_id = p
                break

        try:
            self.panel.set_plugins(items, select_id=select_id)
        except Exception:
            pass

        try:
            self._load_active_plugin_ui()
        except Exception:
            pass

        if getattr(self.pm, "load_errors", None):
            if self.pm.load_errors:
                self.set_status("Some plugins failed to load (see console).", 8000, "warn")
                for e in self.pm.load_errors:
                    print("[PLUGIN-ERROR]\n", e)

    def reload_plugins(self):
        # Re-scan plugin folder and refresh the dropdown/UI
        try:
            self.pm.scan()
        except Exception:
            pass

        # safety limits (global + per-plugin overrides)
        try:
            self._global_limits = dict(GLOBAL_LIMITS)
        except Exception:
            pass

        self._populate_plugins(initial=False)
        self.set_status("Plugins reloaded.", 3000, "info")


    def _on_plugin_changed(self, plugin_id: str):
        """Handle plugin selection change from the left panel."""
        pid = str(plugin_id or "")
        if not pid:
            try:
                pid = str(self.panel.active_plugin_id() or "")
            except Exception:
                pid = ""

        # Apply preset first (N/BOUNDS/ISO) if enabled
        try:
            self._apply_plugin_preset(pid)
        except Exception:
            pass

        # Rebuild the plugin-specific UI
        try:
            self._load_active_plugin_ui()
        except Exception as e:
            try:
                self.set_status(f"Plugin UI load failed: {e!r}", 8000, "err")
            except Exception:
                pass
            return

        # Friendly status
        try:
            info = self.pm.get(pid)
            name = info.name if info is not None else pid
            self.set_status(f"Plugin: {name}", 2000, "info")
        except Exception:
            pass

        # Important: trigger recompute
        try:
            self.recompute()
        except Exception as e:
            try:
                self.set_status(f"Recompute failed: {e!r}", 8000, "err")
            except Exception:
                pass


    def _load_active_plugin_ui(self):
        pid = self.panel.active_plugin_id()
        info = self.pm.get(pid)
        if info is None:
            return

        mod = info.module
        defaults = {}
        try:
            lim = self._get_plugin_limits(pid)
            self.panel.set_n_max(int(lim.get('MAX_N', self._global_limits.get('MAX_N', 520))))
        except Exception:
            pass
        try:
            defaults = dict(mod.get_defaults() or {})
        except Exception:
            defaults = {}

        w, getter = mod.build_ui(self.panel)
        self.panel.set_plugin_ui(w, getter, defaults)
        try:
            self._update_formulas_panel()
        except Exception:
            pass



    def _apply_plugin_preset(self, plugin_id: str) -> None:
        """Apply saved preset (N/BOUNDS/ISO) for the given plugin, if enabled."""
        try:
            if not bool(self.panel.chkUsePresets.isChecked()):
                return
        except Exception:
            return

        pid = str(plugin_id or self.panel.active_plugin_id() or "")
        if not pid:
            return

        preset = get_plugin_preset(pid)
        if not preset:
            return

        # Apply values safely (respect cycle max N if auto-cycle is on)
        try:
            if "N" in preset:
                n = int(preset["N"])
                if getattr(self, "_cycle_on", False):
                    n = min(n, int(getattr(self, "_cycle_max_n", n)))
                # also respect UI max
                try:
                    n = min(n, int(self.panel.spinN.maximum()))
                except Exception:
                    pass
                self.panel.spinN.setValue(n)
        except Exception:
            pass

        try:
            if "BOUNDS" in preset:
                self.panel.dblBounds.setValue(float(preset["BOUNDS"]))
        except Exception:
            pass

        try:
            if "ISO" in preset:
                self.panel.dblIso.setValue(float(preset["ISO"]))
        except Exception:
            pass


    def _on_save_preset(self) -> None:
        """Save current N/BOUNDS/ISO as preset for the active plugin."""
        try:
            pid = str(self.panel.active_plugin_id() or "")
            if not pid:
                return
            preset = {
                "N": int(self.panel.spinN.value()),
                "BOUNDS": float(self.panel.dblBounds.value()),
                "ISO": float(self.panel.dblIso.value()),
            }
            set_plugin_preset(pid, preset)
        except Exception:
            pass


    def _on_clear_preset(self) -> None:
        """Remove saved preset for the active plugin."""
        try:
            pid = str(self.panel.active_plugin_id() or "")
            if not pid:
                return
            clear_plugin_preset(pid)
        except Exception:
            pass

    def _active_plugin_module(self):
        pid = self.panel.active_plugin_id()
        info = self.pm.get(pid)
        return None if info is None else info.module

    def _get_plugin_limits(self, plugin_id: str) -> dict:
        lim = dict(self._global_limits)
        try:
            pl = PLUGIN_LIMITS.get(str(plugin_id), None)
            if isinstance(pl, dict):
                lim.update(pl)
        except Exception:
            pass
        return lim

    def _update_formulas_panel(self):
        # Prefer plugin formula first, allow registry override (when not just a TODO stub)
        try:
            pid = str(self.panel.active_plugin_id() or "")
            mod = self._active_plugin_module()
            txt = None

            # --- plugin-provided ---
            if mod is not None:
                if hasattr(mod, "get_formula"):
                    try:
                        txt = mod.get_formula()
                    except Exception:
                        txt = None
                if not txt and hasattr(mod, "FORMULA"):
                    try:
                        txt = str(getattr(mod, "FORMULA"))
                    except Exception:
                        txt = None

            # --- registry override (ignore auto-generated TODO stubs) ---
            reg_txt = None
            if pid:
                reg_txt = FORMULAS.get(pid)
            if reg_txt:
                s = str(reg_txt).strip()
                is_todo_stub = ("\n- TODO" in s) and (len(s) < 300)
                if not is_todo_stub:
                    txt = reg_txt

            if not txt:
                name = pid
                try:
                    name = getattr(mod, "PLUGIN_META", {}).get("name", pid) if mod is not None else pid
                except Exception:
                    pass
                txt = f"{name}\n\n(This plugin currently does not have formulas v registri.\n" \
                      f"Add an override in core/formulas_registry.py -> FORMULAS['{pid}'].)"

            # --- sanitize output: remove leading blank rows / stray backslashes ---
            try:
                t = str(txt).replace("\r\n", "\n").replace("\r", "\n")
                t = "\n".join([ln for ln in t.split("\n") if ln.strip() != "\\"])
                t = t.strip("\n")
                txt = t
            except Exception:
                pass

            try:
                self._formulas_text.setPlainText(txt)
            except Exception:
                pass
        except Exception as e:
            try:
                self._formulas_text.setPlainText(f"Formulas: error\n{e!r}")
            except Exception:
                pass


    # =========================
    # Menu
    # =========================
    def _build_menu(self):
        mb = self.win.menuBar()
        mb.clear()

        menu_file = mb.addMenu("File")
        menu_render = mb.addMenu("Render")
        menu_color = mb.addMenu("Color")
        menu_view = mb.addMenu("View")
        menu_workbench = mb.addMenu("Workbench")
        menu_help = mb.addMenu("Help")

        actShotUi = QtGui.QAction("Export UI screenshot…", self.win)
        actShotUi.triggered.connect(self.export_ui_screenshot)

        actShotClean = QtGui.QAction("Export clean render…", self.win)
        actShotClean.setShortcut("S")
        actShotClean.triggered.connect(self.screenshot)

        actStl = QtGui.QAction("Export current STL…", self.win)
        actStl.triggered.connect(self.export_current_stl)

        actPack = QtGui.QAction("Export all pack…", self.win)
        actPack.triggered.connect(self.export_all_pack_dialog)

        actQuit = QtGui.QAction("Quit", self.win)
        actQuit.setShortcut("Ctrl+Q")
        actQuit.triggered.connect(self.win.close)

        menu_file.addAction(actShotUi)
        menu_file.addAction(actShotClean)
        menu_file.addAction(actStl)
        menu_file.addSeparator()
        menu_file.addAction(actPack)
        menu_file.addSeparator()
        menu_file.addAction(actQuit)

        self.actSmooth = QtGui.QAction("Smooth shading", self.win, checkable=True)
        self.actSmooth.setChecked(bool(self.state["SMOOTH_SHADING"]))
        self.actSmooth.triggered.connect(self._toggle_smooth)

        self.actAA = QtGui.QAction("Anti-alias (FXAA)", self.win, checkable=True)
        self.actAA.setChecked(bool(self.state["ANTIALIAS"]))
        self.actAA.triggered.connect(self._toggle_aa)

        menu_render.addAction(self.actSmooth)
        menu_render.addAction(self.actAA)

        self.grpColorMode = QtGui.QActionGroup(self.win)
        self.grpColorMode.setExclusive(True)
        menu_color.addSection("Color by")

        def add_mode(title: str, key: str):
            act = QtGui.QAction(title, self.win, checkable=True)
            act.setData(key)
            act.setActionGroup(self.grpColorMode)
            if self.state["COLOR_MODE"] == key:
                act.setChecked(True)
            menu_color.addAction(act)

        add_mode("Field value", "value")
        add_mode("Height (Z)", "height")
        add_mode("Radius", "radius")
        self.grpColorMode.triggered.connect(self._set_color_mode)

        menu_color.addSeparator()
        self.actScalarBar = QtGui.QAction("Show scalar bar", self.win, checkable=True)
        self.actScalarBar.setChecked(bool(self.state["SHOW_SCALAR_BAR"]))
        self.actScalarBar.triggered.connect(self._toggle_scalar_bar)
        menu_color.addAction(self.actScalarBar)
        menu_color.addSeparator()

        cmap_menu = menu_color.addMenu("Colormap")
        cmaps = ["turbo", "viridis", "plasma", "inferno", "magma", "cividis", "hsv", "gist_rainbow"]
        self.grpCmap = QtGui.QActionGroup(self.win)
        self.grpCmap.setExclusive(True)
        current = self.state["COLORMAP_DARK"] if self.dark_bg else self.state["COLORMAP_LIGHT"]
        for name in cmaps:
            act = QtGui.QAction(name, self.win, checkable=True)
            act.setData(name)
            act.setActionGroup(self.grpCmap)
            if name == current:
                act.setChecked(True)
            cmap_menu.addAction(act)
        self.grpCmap.triggered.connect(self._set_colormap)

        self.actAxes = QtGui.QAction("Axes", self.win, checkable=True)
        self.actAxes.setChecked(bool(self.state["SHOW_AXES"]))
        self.actAxes.triggered.connect(self.toggle_axes)

        self.actBounds = QtGui.QAction("Bounds box", self.win, checkable=True)
        self.actBounds.setChecked(bool(self.state.get("SHOW_BOUNDS", False)))
        self.actBounds.triggered.connect(self.toggle_bounds)

        self.actDark = QtGui.QAction("Dark background", self.win, checkable=True)
        self.actDark.setChecked(bool(self.state["DARK_BG"]))
        self.actDark.triggered.connect(self.toggle_bg)

        self.actShowPanel = QtGui.QAction("Show left panel", self.win, checkable=True)
        self.actShowPanel.setChecked(True)
        self.actShowPanel.triggered.connect(lambda checked: self._dock.setVisible(checked))

        menu_view.addAction(self.actAxes)
        menu_view.addAction(self.actBounds)
        menu_view.addAction(self.actDark)
        menu_view.addAction(self.actShowPanel)

        self.actFormulas = QtGui.QAction("Show formulas panel", self.win, checkable=True)
        self.actFormulas.setChecked(False)
        self.actFormulas.triggered.connect(lambda checked: self._formulas_dock.setVisible(bool(checked)))
        menu_view.addAction(self.actFormulas)
        menu_view.addSeparator()

        self.actFullscreen = QtGui.QAction("Fullscreen", self.win, checkable=True)
        self.actFullscreen.setShortcut("F11")
        self.actFullscreen.triggered.connect(self.toggle_fullscreen)
        menu_view.addAction(self.actFullscreen)

        self.actAutopilot = QtGui.QAction("Autopilot", self.win, checkable=True)
        self.actAutopilot.setShortcut("Space")
        self.actAutopilot.triggered.connect(lambda checked: self.toggle_autopilot(checked))
        menu_view.addAction(self.actAutopilot)

        self.actCycle = QtGui.QAction("Auto-cycle plugins", self.win, checkable=True)
        self.actCycle.setShortcut("C")
        self.actCycle.triggered.connect(lambda checked: self.toggle_cycle_plugins(checked))
        menu_view.addAction(self.actCycle)

        actRecompute = QtGui.QAction("Recompute", self.win)
        actRecompute.setShortcut("Ctrl+Enter")
        actRecompute.triggered.connect(self.recompute)

        actReset = QtGui.QAction("Reset view", self.win)
        actReset.setShortcut("R")
        actReset.triggered.connect(self.reset_view)

        actReload = QtGui.QAction("Reload plugins", self.win)
        actReload.setShortcut("Ctrl+R")
        actReload.triggered.connect(self.reload_plugins)

        menu_workbench.addAction(actRecompute)
        menu_workbench.addSeparator()
        menu_workbench.addAction(actReset)
        menu_workbench.addAction(actReload)

        actAbout = QtGui.QAction("About…", self.win)
        actAbout.triggered.connect(self._show_about)
        menu_help.addAction(actAbout)

    def _show_about(self):
        dlg = QtWidgets.QDialog(self.win)
        dlg.setWindowTitle("About FieldForge3D")
        dlg.setModal(True)
        dlg.setMinimumWidth(460)

        lay = QtWidgets.QVBoxLayout(dlg)
        txt = QtWidgets.QLabel(
            f"<b>FieldForge3D</b><br>"
            f"Version: {APP_VERSION}<br>"
            "Author: Tibor Čefan (finky666)<br>"
            "Refactor & export pipeline: ChatGPT (Majka / SuPyWomen)<br><br>"
            "Interactive 3D implicit-field playground with export workflow for showcase, screensaver and 3D print.<br>"
            "Version 1.1 adds UI cleanup and a new hidden bloom plugin.<br><br>"
            "PyQt6 + NumPy + Numba + PyVista/VTK"
        )
        txt.setTextFormat(QtCore.Qt.TextFormat.RichText)
        txt.setWordWrap(True)
        lay.addWidget(txt)

        secret = QtWidgets.QLabel('<a href="egg" style="color:#666666; text-decoration:none;">.</a>', dlg)
        secret.setTextFormat(QtCore.Qt.TextFormat.RichText)
        secret.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextBrowserInteraction)
        secret.setOpenExternalLinks(False)
        secret.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        secret.setToolTip("Easter egg")
        lay.addWidget(secret)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok, parent=dlg)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)

        def _on_secret(_link: str):
            dlg.accept()
            self._activate_hidden_plugin("event_horizon_bloom")

        secret.linkActivated.connect(_on_secret)
        dlg.exec()

    def _activate_hidden_plugin(self, plugin_id: str):
        info = self.pm.get(plugin_id)
        if info is None:
            self.set_status("Hidden plugin not found.", 4000, "warn")
            return

        cmb = self.panel.cmbPlugin
        idx = cmb.findData(plugin_id)
        if idx < 0:
            cmb.addItem(info.name, plugin_id)
            idx = cmb.findData(plugin_id)
        if idx >= 0:
            cmb.setCurrentIndex(idx)
            self.set_status(f"Hidden plugin unlocked: {info.name}", 3000, "info")

    # =========================
    # Status helpers
    # =========================
    # Status helpers
    # =========================
    def _set_status_label(self, text: str):
        txt = (text or "").replace("\n", " ").strip().strip()
        fm = self._status_label.fontMetrics()
        w = max(80, int(self.status.width()) - 20)
        self._status_label.setText(fm.elidedText(txt, QtCore.Qt.TextElideMode.ElideRight, w))

    def set_status(self, msg: str, timeout_ms: int = 0, level: str = "info", sticky: bool = False):
        if sticky:
            timeout_ms = 0

        if sticky and level in ("warn", "err"):
            self._status_lock = True
        elif level == "info":
            pass
        else:
            self._status_lock = False

        if level == "warn":
            self.status.setStyleSheet("QStatusBar{color:#ffcc66;}")
        elif level == "err":
            self.status.setStyleSheet("QStatusBar{color:#ff6666;}")
        else:
            self.status.setStyleSheet("QStatusBar{color:#dddddd;}")

        self._set_status_label(msg or "")

        if (not sticky) and int(timeout_ms) > 0:
            QtCore.QTimer.singleShot(int(timeout_ms), self._restore_idle_status)

    def _restore_idle_status(self):
        if getattr(self, "_status_lock", False):
            return
        try:
            self.status.setStyleSheet("QStatusBar{color:#dddddd;}")
            self._set_status_label(self._idle_status_text)
        except Exception:
            pass

    # =========================
    # Safe picking patch
    # =========================
    def _patch_safe_picking(self):
        orig = self.plotter.enable_cell_picking

        def safe_enable_cell_picking(*args, **kwargs):
            try:
                self.plotter.disable_picking()
            except Exception:
                pass
            return orig(*args, **kwargs)

        self.plotter.enable_cell_picking = safe_enable_cell_picking

    # =========================
    # View / Render
    # =========================
    def _apply_view_options(self):
        self.plotter.set_background("black" if self.dark_bg else "white")

        if self.axes_visible:
            self.plotter.show_axes()
        else:
            self.plotter.hide_axes()

        # Optional bounds box + numeric axes (can confuse users; keep OFF by default)
        if bool(self.state.get("SHOW_BOUNDS", False)):
            try:
                self.plotter.show_bounds(grid=False, location="outer")
            except Exception:
                pass
        else:
            try:
                self.plotter.remove_bounds_axes()
            except Exception:
                pass

        if self.state["ANTIALIAS"]:
            self.plotter.enable_anti_aliasing("fxaa")
        else:
            try:
                self.plotter.disable_anti_aliasing()
            except Exception:
                pass

        self.plotter.render()

    def _toggle_smooth(self, checked: bool):
        self.state["SMOOTH_SHADING"] = bool(checked)
        self._recolor_only()

    def _toggle_aa(self, checked: bool):
        self.state["ANTIALIAS"] = bool(checked)
        self._apply_view_options()

    def _toggle_scalar_bar(self, checked: bool):
        self.state["SHOW_SCALAR_BAR"] = bool(checked)
        self._recolor_only()

    def _set_color_mode(self, act: QtGui.QAction):
        self.state["COLOR_MODE"] = str(act.data())
        self._recolor_only()

    def _set_colormap(self, act: QtGui.QAction):
        cmap = str(act.data())
        if self.dark_bg:
            self.state["COLORMAP_DARK"] = cmap
        else:
            self.state["COLORMAP_LIGHT"] = cmap
        self._recolor_only()

    def toggle_axes(self):
        self.axes_visible = not self.axes_visible
        self.state["SHOW_AXES"] = self.axes_visible
        self.actAxes.setChecked(self.axes_visible)
        self._apply_view_options()

    def toggle_bounds(self):
        cur = bool(self.state.get("SHOW_BOUNDS", False))
        cur = not cur
        self.state["SHOW_BOUNDS"] = cur
        if hasattr(self, "actBounds"):
            self.actBounds.setChecked(cur)
        self._apply_view_options()

    def toggle_bg(self):
        self.dark_bg = not self.dark_bg
        self.state["DARK_BG"] = self.dark_bg
        self.actDark.setChecked(self.dark_bg)

        current = self.state["COLORMAP_DARK"] if self.dark_bg else self.state["COLORMAP_LIGHT"]
        for act in self.grpCmap.actions():
            if act.data() == current:
                act.setChecked(True)
                break

        self._apply_view_options()
        self._recolor_only()

    def reset_view(self):
        self.plotter.reset_camera()
        self.plotter.render()

    def screenshot(self):
        fname = time.strftime("field_%Y%m%d_%H%M%S.png")
        self.plotter.screenshot(fname)
        print(f"[INFO] Screenshot: {os.path.abspath(fname)}")
        self.set_status(f"Clean render saved: {fname}", 6000, "info")

    def export_ui_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.win, "Export UI screenshot", "fieldforge_ui.png", "PNG Image (*.png)"
        )
        if not path:
            return
        self.win.grab().save(path)
        self.set_status(f"UI screenshot saved: {Path(path).name}", 6000, "info")

    def export_current_stl(self):
        if self._last_surf is None or self._last_surf.n_points == 0:
            self.set_status("No current mesh to export.", 4000, "warn")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.win, "Export current STL", "fieldforge_current.stl", "STL Mesh (*.stl)"
        )
        if not path:
            return
        export_stl(self._last_surf, path, max_size_mm=100.0)
        self.set_status(f"STL exported: {Path(path).name}", 6000, "info")

    def export_all_pack_dialog(self):
        base = QtWidgets.QFileDialog.getExistingDirectory(self.win, "Export FieldForge3D pack")
        if not base:
            return
        out_dir = Path(base) / f"fieldforge_export_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        current_ui = out_dir / "current_ui.png"
        current_clean = out_dir / "current_clean.png"
        self.win.grab().save(str(current_ui))
        if self._last_surf is not None and self._last_surf.n_points > 0:
            self.plotter.screenshot(str(current_clean))

        state = {
            "DARK_BG": self.dark_bg,
            "SMOOTH_SHADING": self.state["SMOOTH_SHADING"],
            "COLOR_MODE": self.state["COLOR_MODE"],
            "COLORMAP_DARK": self.state["COLORMAP_DARK"],
            "COLORMAP_LIGHT": self.state["COLORMAP_LIGHT"],
            "SHOW_SCALAR_BAR": self.state["SHOW_SCALAR_BAR"],
        }
        manifest = export_all_pack(self.plugins_dir, out_dir, state=state)
        count_ok = sum(1 for rec in manifest["records"] if not rec.get("skipped"))
        self.set_status(f"Export pack ready: {count_ok} items -> {out_dir.name}", 8000, "info")

    # =========================
    # Recompute pipeline
    # =========================
    def _is_worker_running(self) -> bool:
        """Return True if a compute worker thread is currently running (crash-safe)."""
        th = getattr(self, "_thread", None)
        if th is None:
            return False
        try:
            return bool(th.isRunning())
        except RuntimeError:
            return False
        except Exception:
            return False

    def _maybe_run_pending_recompute(self):
        """Run one queued recompute after current compute finishes."""
        if not getattr(self, "_pending_recompute", False):
            return
        self._pending_recompute = False
        self._pending_reason = ""
        try:
            self.set_status("Running queued recompute…", 2000, "info")
        except Exception:
            pass
        QtCore.QTimer.singleShot(0, self.recompute)

    def recompute(self):
        # If a compute is already running, don't drop the request – queue exactly one recompute.
        if self._is_worker_running():
            self._pending_recompute = True
            self._pending_reason = "busy"
            self.set_status("Compute running… queued next recompute.", 1500, "warn")
            return

        if self._thread is not None and self._thread.isRunning():
            print("[WARN] A computation is already running – please wait.")
            return

        plugin = self._active_plugin_module()
        if plugin is None:
            self.set_status("No active plugin.", 0, "err", sticky=True)
            return

        params = dict(self.panel.get_params())

        # --- safety: clamp N by global + per-plugin limits ---
        pid = str(self.panel.active_plugin_id() or "")
        lim = self._get_plugin_limits(pid)
        try:
            n_in = int(params.get("N", 0))
        except Exception:
            n_in = 0
        max_n = int(lim.get("MAX_N", self._global_limits.get("MAX_N", 520)))
        if n_in > max_n:
            params["N"] = max_n
            try:
                self.panel.spinN.setValue(max_n)
            except Exception:
                pass
            self.set_status(f"Safety: N reduced to {max_n} (plugin limit).", 4000, "warn")

        voxels = int(params["N"]) ** 3
        print(f"[INFO] RECOMPUTE: plugin={self.panel.active_plugin_id()} | N={params['N']} -> {voxels:,} voxelov")

        # --- global 
        # --- param safety clamps (CPU/stability) ---
        params, clamps = clamp_params(pid, params)
        if clamps:
            try:
                msg = ", ".join([f"{k}:{old}->{new}" for (k, old, new) in clamps[:3]])
                if len(clamps) > 3:
                    msg += f" (+{len(clamps)-3} more)"
                self.set_status(f"[SAFE] Clamped: {msg}", 6000, "warn")
            except Exception:
                pass

        # --- Memory Guard (applies to ALL plugins) ---
        try:
            meta = getattr(plugin, "PLUGIN_META", {}) or {}
        except Exception:
            meta = {}

        extra_buffers = None
        try:
            extra_buffers = meta.get("mem_extra_buffers", None) if isinstance(meta, dict) else None
        except Exception:
            extra_buffers = None

        # --- global Memory Guard (collision-only) ---
        # Policy requested:
        #   - don't bother user at startup
        #   - don't spam yellow; warn only if it's truly unsafe ("red")
        #   - if user already accepted Continue for (plugin_id, N) in this session, don't ask again

        if getattr(self, "_suppress_guard_once", False):
            # first render should always be smooth
            self._suppress_guard_once = False
            self.panel.clear_guard()
        else:
            est = mem_guard(int(params["N"]), extra_buffers=extra_buffers)

            if est.level != "red":
                # no UI noise in green/yellow
                self.panel.clear_guard()
            else:
                badge_text = (
                    f"RAM estimate ~{est.est_gb:.2f} GB, available {est.avail_gb:.2f} GB "
                    f"({est.ratio*100:.1f}% of available)."
                )

                # plugin-specific warnings hook (optional) – only when red
                try:
                    get_warnings = getattr(plugin, "get_warnings", None)
                    if callable(get_warnings):
                        w = get_warnings(dict(params), est)
                        if isinstance(w, (list, tuple)) and w:
                            badge_text = badge_text + "  " + "  ".join(str(x) for x in w[:3])
                except Exception:
                    pass

                self.panel.set_guard("red", badge_text)

                # already acknowledged for this session?
                key = (pid, int(params["N"]))
                if key not in getattr(self, "_mem_guard_ack", set()):
                    mb = QtWidgets.QMessageBox(self.win)
                    mb.setWindowTitle("Memory Guard")
                    mb.setIcon(QtWidgets.QMessageBox.Icon.Critical)

                    mb.setText(
                        f"Plugin: {pid}\n"
                        f"N = {est.N}  (voxels: {est.voxels:,})\n\n"
                        f"Estimated RAM: {est.est_gb:.2f} GB\n"
                        f"Available RAM: {est.avail_gb:.2f} GB\n"
                        f"Ratio: {est.ratio*100:.1f}% of available\n\n"
                        f"Verdict: RED – {est.note}"
                    )

                    btn_continue = mb.addButton("Continue", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                    btn_auto = mb.addButton("Auto-reduce N", QtWidgets.QMessageBox.ButtonRole.ActionRole)
                    btn_cancel = mb.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)

                    mb.exec()
                    clicked = mb.clickedButton()

                    if clicked == btn_cancel:
                        self.set_status("Cancelled.", 2000, "warn")
                        self.panel.set_busy(False)
                        return

                    if clicked == btn_auto:
                        newN = suggest_lower_N(int(params["N"]), target_ratio=0.40, current_ratio=float(est.ratio))
                        if newN < int(params["N"]):
                            params["N"] = int(newN)
                            try:
                                self.panel.spinN.setValue(int(newN))
                            except Exception:
                                pass
                            voxels2 = int(params["N"]) ** 3
                            print(f"[INFO] Memory Guard: auto-reduced N -> {params['N']} ({voxels2:,} voxels)")
                            est2 = mem_guard(int(params["N"]), extra_buffers=extra_buffers)
                            if est2.level == "red":
                                self.panel.set_guard("red", f"Auto-reduced N but still RED. RAM estimate ~{est2.est_gb:.2f} GB, available {est2.avail_gb:.2f} GB ({est2.ratio*100:.1f}%).")
                            else:
                                self.panel.clear_guard()

                    # Remember: user accepted to continue (or auto-reduced and still wants to try)
                    try:
                        self._mem_guard_ack.add((pid, int(params["N"])))
                    except Exception:
                        pass


        self.panel.set_busy(True)
        self._status_lock = False
        self.set_status("Computing…", 0, "info", sticky=True)

        self._thread = QtCore.QThread()
        self._worker = ComputeWorker(plugin, params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_field_ready)
        self._worker.failed.connect(self._on_worker_fail)

        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)

        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._worker.deleteLater)

        self._thread.start()

    @QtCore.pyqtSlot(object, dict, float)
    def _on_field_ready(self, field, p, field_s: float):
        try:
            print(f"[INFO] Field computed in {field_s:.2f}s")

            N = int(p["N"])
            bounds = float(p["BOUNDS"])
            iso = float(p["ISO"])

            spacing = (2 * bounds / (N - 1), 2 * bounds / (N - 1), 2 * bounds / (N - 1))
            origin = (-bounds, -bounds, -bounds)

            t1 = time.time()
            grid = pv.ImageData(dimensions=(N, N, N), spacing=spacing, origin=origin)

            arr = np.asarray(field, dtype=np.float32)
            grid["value"] = arr.ravel(order="F")

            surf = grid.contour(
                isosurfaces=[iso],
                scalars="value",
                compute_normals=False,
                compute_gradients=False,
            )

            t2 = time.time()
            print(f"[INFO] Contour in {t2 - t1:.2f}s | points={surf.n_points:,} | polys={surf.n_cells:,}")

            # empty guard
            if surf.n_points == 0 or surf.n_cells == 0:
                msg = "Iso-surface is empty (0 points). Try a different ISO or parameters."
                print("[WARN]", msg)
                self.set_status(msg, 0, "warn", sticky=True)
                self.plotter.render()
                return

            # ---- SAFETY: mesh density gate (avoid VTK crashes) ----
            polys = int(surf.n_cells)

            pid = str(self.panel.active_plugin_id() or "")
            lim = self._get_plugin_limits(pid)
            HARD_STOP = int(lim.get("HARD_STOP", self._global_limits.get("HARD_STOP", 18_000_000)))
            MAX_POLYS = int(lim.get("MAX_POLYS", self._global_limits.get("MAX_POLYS", 4_000_000)))

            if polys > HARD_STOP:
                msg = f"Mesh extremely dense ({polys:,} polys). Stopped (safety guard)."
                print("[ERROR]", msg)
                self.set_status(msg, 0, "err", sticky=True)
                return

            if polys > MAX_POLYS:
                msg = f"Mesh too dense ({polys:,} polys). Decimating…"
                print("[WARN]", msg)
                self.set_status(msg, 6000, "warn")

                target = float(MAX_POLYS)
                p = float(polys)
                reduction = 1.0 - (target / p)
                if reduction < 0.10:
                    reduction = 0.10
                if reduction > 0.95:
                    reduction = 0.95
                try:
                    print(f"[WARN] Decimate reduction={reduction:.2f}")
                    surf = surf.decimate_pro(reduction)
                    surf = surf.clean()
                    print(f"[INFO] After decimate: points={surf.n_points:,} | polys={surf.n_cells:,}")
                except Exception as e:
                    emsg = f"Decimate zlyhalo: {e!r}"
                    print("[ERROR]", emsg)
                    self.set_status(emsg, 0, "err", sticky=True)
                    return

            self.set_status(f"Contour: points={surf.n_points:,} | polys={surf.n_cells:,}", 5000, "info")

            # ensure scalars exist
            if "value" not in surf.array_names:
                surf["value"] = np.zeros(surf.n_points, dtype=np.float32)

            # extra scalars
            surf["height"] = surf.points[:, 2].astype(np.float32)
            surf["radius"] = np.linalg.norm(surf.points, axis=1).astype(np.float32)

            self._last_surf = surf

            try:
                self.plotter.remove_actor("mesh")
            except Exception:
                pass

            self._add_mesh(surf)

            # lights
            self.plotter.remove_all_lights()
            key = pv.Light(position=(3, 2, 4), focal_point=(0, 0, 0), intensity=1.0, light_type="scene light")
            fill = pv.Light(position=(-3, -2, 2), focal_point=(0, 0, 0), intensity=0.55, light_type="scene light")
            rim = pv.Light(position=(0, 5, -2), focal_point=(0, 0, 0), intensity=0.35, light_type="scene light")
            self.plotter.add_light(key)
            self.plotter.add_light(fill)
            self.plotter.add_light(rim)

            # always fit camera to the newly generated object
            self.plotter.reset_camera()
            self._first_render_done = True

            self._apply_view_options()
            self.plotter.render()
            print("[DONE]")

        finally:
            self.panel.set_busy(False)
            if not getattr(self, "_status_lock", False):
                self.set_status("Done.", 1500, "info", sticky=False)
            self._thread = None
            self._worker = None
            try:
                self._maybe_run_pending_recompute()
            except Exception:
                pass

    def _add_mesh(self, surf: pv.PolyData):
        mode = self.state["COLOR_MODE"]
        cmap = self.state["COLORMAP_DARK"] if self.dark_bg else self.state["COLORMAP_LIGHT"]
        show_bar = bool(self.state["SHOW_SCALAR_BAR"])

        scalar_map = {"value": "value", "height": "height", "radius": "radius"}
        scalars = scalar_map.get(mode, "value")

        scalar_bar_args = dict(title=scalars, n_labels=5, shadow=True, fmt="%.2f")

        self.plotter.add_mesh(
            surf,
            name="mesh",
            scalars=scalars,
            cmap=cmap,
            show_scalar_bar=show_bar,
            scalar_bar_args=scalar_bar_args if show_bar else None,
            smooth_shading=bool(self.state["SMOOTH_SHADING"]),
            specular=0.55,
            specular_power=55,
            diffuse=0.8,
            ambient=0.2,
        )

    def _recolor_only(self):
        if self._last_surf is None:
            self.plotter.render()
            return

        try:
            self.plotter.remove_actor("mesh")
        except Exception:
            pass

        self._add_mesh(self._last_surf)
        self.plotter.render()

    @QtCore.pyqtSlot(str)
    def _on_worker_fail(self, msg: str):
        try:
            print("[ERROR]", msg)
            self.set_status(f"Chyba: {msg}", 0, "err", sticky=True)
            QtWidgets.QMessageBox.critical(self.win, "Computation error", msg)
        finally:
            self.panel.set_busy(False)
            self._thread = None
            self._worker = None
            try:
                self._maybe_run_pending_recompute()
            except Exception:
                pass

    # =========================
    # Autopilot / Fullscreen / Cycle
    # =========================
    def toggle_fullscreen(self):
        self._is_fullscreen = not getattr(self, "_is_fullscreen", False)
        if self._is_fullscreen:
            self.win.showFullScreen()
        else:
            self.win.showNormal()

        if hasattr(self, "actFullscreen"):
            self.actFullscreen.setChecked(self._is_fullscreen)

    def toggle_autopilot(self, checked: bool | None = None):
        if checked is None:
            self._autopilot_on = not getattr(self, "_autopilot_on", False)
        else:
            self._autopilot_on = bool(checked)

        if hasattr(self, "actAutopilot"):
            self.actAutopilot.setChecked(self._autopilot_on)

        self._autopilot_phase = 0.0

    def toggle_cycle_plugins(self, checked: bool | None = None):
        if checked is None:
            self._cycle_on = not getattr(self, "_cycle_on", False)
        else:
            self._cycle_on = bool(checked)

        if hasattr(self, "actCycle"):
            self.actCycle.setChecked(self._cycle_on)

        if self._cycle_on:
            try:
                if self.panel.spinN.value() > self._cycle_max_n:
                    self.panel.spinN.setValue(self._cycle_max_n)
            except Exception:
                pass

        self._cycle_next_t = float(getattr(self, "_autopilot_t", 0.0)) + float(self._cycle_seconds)

    def _on_auto_tick(self):
        if not hasattr(self, "_auto_elapsed"):
            return

        dt = self._auto_elapsed.restart() / 1000.0
        if dt <= 0.0:
            dt = 0.033

        self._autopilot_t += dt

        # ----- autopilot camera -----
        if getattr(self, "_autopilot_on", False):
            try:
                deg = self._autopilot_speed_deg * dt
                self.plotter.camera.Azimuth(deg)

                self._autopilot_phase += dt
                breathe = 1.0 + self._autopilot_breathe * math.sin(
                    2.0 * math.pi * (self._autopilot_phase / 6.0)
                )
                self.plotter.camera.Dolly(breathe)

                self.plotter.render()
            except Exception:
                pass

        # ----- cycle plugins -----
        if getattr(self, "_cycle_on", False):
            if self._autopilot_t >= self._cycle_next_t:
                self._cycle_next_t = self._autopilot_t + float(self._cycle_seconds)
                self._cycle_to_next_plugin()

    def _cycle_to_next_plugin(self):
        if self._thread is not None and self._thread.isRunning():
            return

        try:
            cmb = self.panel.cmbPlugin
            n = cmb.count()
            if n <= 1:
                return

            i = cmb.currentIndex()

            for _ in range(2 * n):
                i = (i + 1) % n
                pid = str(cmb.itemData(i) or "")
                if pid and (pid not in self._cycle_blacklist):
                    cmb.setCurrentIndex(i)
                    return

        except Exception:
            pass