# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# core/panel.py
from __future__ import annotations

from PyQt6 import QtWidgets, QtCore


class PluginHostPanel(QtWidgets.QWidget):
    """UI panel (English)."""

    recompute_clicked = QtCore.pyqtSignal()
    reload_plugins_clicked = QtCore.pyqtSignal()
    plugin_changed = QtCore.pyqtSignal(str)
    save_preset_clicked = QtCore.pyqtSignal()
    clear_preset_clicked = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(360)

        self._plugin_get_params = None
        self._plugin_defaults = {}

        # ------------------------------
        # Header
        # ------------------------------
        title = QtWidgets.QLabel("FieldForge3D parameters")
        title.setStyleSheet("font-weight:bold; font-size:13px;")

        # ------------------------------
        # Global guard badge (memory/perf warnings)
        # ------------------------------
        self.lblGuard = QtWidgets.QLabel("")
        self.lblGuard.setWordWrap(True)
        self.lblGuard.setVisible(False)
        self.lblGuard.setStyleSheet("padding:6px; border-radius:8px; font-size:12px;")
        

        # ------------------------------
        # Common controls
        # ------------------------------
        self.spinN = QtWidgets.QSpinBox()
        self.spinN.setRange(60, 520)
        self.spinN.setValue(220)

        self.dblBounds = QtWidgets.QDoubleSpinBox()
        self.dblBounds.setRange(0.25, 5.0)
        self.dblBounds.setSingleStep(0.1)
        self.dblBounds.setDecimals(2)
        self.dblBounds.setValue(1.50)

        self.dblIso = QtWidgets.QDoubleSpinBox()
        self.dblIso.setRange(0.01, 0.99)
        self.dblIso.setSingleStep(0.05)
        self.dblIso.setDecimals(2)
        self.dblIso.setValue(0.60)

        # ------------------------------
        # Per-plugin presets (N/BOUNDS/ISO)
        # ------------------------------
        self.chkUsePresets = QtWidgets.QCheckBox("Use per-plugin presets for N/BOUNDS/ISO")
        self.chkUsePresets.setChecked(True)

        self.btnSavePreset = QtWidgets.QPushButton("Save preset")
        self.btnClearPreset = QtWidgets.QPushButton("Clear preset")
        self.btnSavePreset.clicked.connect(self.save_preset_clicked.emit)
        self.btnClearPreset.clicked.connect(self.clear_preset_clicked.emit)

        # ------------------------------
        # Plugin select
        # ------------------------------
        self.cmbPlugin = QtWidgets.QComboBox()
        self.cmbPlugin.currentIndexChanged.connect(self._on_plugin_combo_changed)

        # ------------------------------
        # Plugin UI host (scrollable ONLY here)
        # ------------------------------
        self.pluginHost = QtWidgets.QGroupBox("Plugin settings")

        self._plugin_container = QtWidgets.QWidget()
        self.pluginLayout = QtWidgets.QVBoxLayout(self._plugin_container)
        self.pluginLayout.setContentsMargins(8, 8, 8, 8)
        self.pluginLayout.setSpacing(6)

        placeholder = QtWidgets.QLabel("No plugin loaded.")
        placeholder.setStyleSheet("color:#aaaaaa;")
        self.pluginLayout.addWidget(placeholder)
        self.pluginLayout.addStretch(1)

        self.pluginScroll = QtWidgets.QScrollArea()
        self.pluginScroll.setWidgetResizable(True)
        self.pluginScroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.pluginScroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pluginScroll.setWidget(self._plugin_container)

        hostLayout = QtWidgets.QVBoxLayout(self.pluginHost)
        hostLayout.setContentsMargins(6, 6, 6, 6)
        hostLayout.addWidget(self.pluginScroll)

        # Keep plugin area "human sized" by default (so buttons stay visible)
        row_h = self.fontMetrics().height() + 14
        self.pluginScroll.setMinimumHeight(row_h * 5)
        self.pluginScroll.setMaximumHeight(16777215)  # no max

        self.pluginHost.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        # ------------------------------
        # Buttons (always visible)
        # ------------------------------
        self.btnRecompute = QtWidgets.QPushButton("RECOMPUTE")
        self.btnRecompute.setStyleSheet("font-weight:bold; padding:8px;")

        self.btnDefaults = QtWidgets.QPushButton("Reset default")
        self.btnReload = QtWidgets.QPushButton("Reload plugins")

        self.btnRecompute.clicked.connect(self.recompute_clicked.emit)
        self.btnReload.clicked.connect(self.reload_plugins_clicked.emit)
        self.btnDefaults.clicked.connect(self.reset_defaults)

        # NOTE: Status is handled by the main window full-width status bar (QStatusBar).
        # Keep the left panel clean (no second status line here).

        # ------------------------------
        # Layout build
        # ------------------------------
        formPlugin = QtWidgets.QFormLayout()
        formPlugin.addRow("Field (plugin)", self.cmbPlugin)

        formCommon = QtWidgets.QFormLayout()
        formCommon.addRow("N", self.spinN)
        formCommon.addRow("BOUNDS", self.dblBounds)
        formCommon.addRow("ISO", self.dblIso)

        presetsRow = QtWidgets.QHBoxLayout()
        presetsRow.addWidget(self.btnSavePreset)
        presetsRow.addWidget(self.btnClearPreset)

        v = QtWidgets.QVBoxLayout()
        v.addWidget(title)
        v.addWidget(self.lblGuard)
        v.addLayout(formPlugin)
        v.addSpacing(6)
        v.addLayout(formCommon)
        v.addSpacing(6)
        v.addWidget(self.chkUsePresets)
        v.addLayout(presetsRow)
        v.addSpacing(4)

        # Only plugin settings scroll
        v.addWidget(self.pluginHost)
        v.addSpacing(4)

        # Footer: buttons (always visible)
        v.addWidget(self.btnRecompute)
        v.addWidget(self.btnDefaults)
        v.addWidget(self.btnReload)

        # Small bottom spacer
        after_pad = QtWidgets.QWidget()
        after_pad.setFixedHeight(6)
        v.addWidget(after_pad)

        self.setLayout(v)

    # ------------------------------------------------------------
    # Public API used by app.py
    # ------------------------------------------------------------
    def set_plugins(self, plugins: list[tuple[str, str]], select_id: str | None = None):
        """plugins: list of (id, name)"""
        self.cmbPlugin.blockSignals(True)
        self.cmbPlugin.clear()
        for pid, name in plugins:
            self.cmbPlugin.addItem(name, pid)
        self.cmbPlugin.blockSignals(False)

        if select_id is not None:
            for i in range(self.cmbPlugin.count()):
                if self.cmbPlugin.itemData(i) == select_id:
                    self.cmbPlugin.setCurrentIndex(i)
                    break

    def set_n_max(self, max_n: int):
        """Clamp maximum allowed N in the UI (per-plugin safety)."""
        try:
            max_n = int(max_n)
        except Exception:
            return
        if max_n < 10:
            max_n = 10
        self.spinN.setMaximum(max_n)
        if self.spinN.value() > max_n:
            self.spinN.setValue(max_n)

    def active_plugin_id(self) -> str:
        return str(self.cmbPlugin.currentData() or "")

    def set_plugin_ui(self, widget: QtWidgets.QWidget, get_params_callable, plugin_defaults: dict):
        # clear host
        while self.pluginLayout.count():
            item = self.pluginLayout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        # add new UI + stretch so it stays at top
        self.pluginLayout.addWidget(widget)
        self.pluginLayout.addStretch(1)

        self._plugin_get_params = get_params_callable
        self._plugin_defaults = dict(plugin_defaults or {})

        # scroll to top after plugin change
        QtCore.QTimer.singleShot(0, lambda: self.pluginScroll.verticalScrollBar().setValue(0))

    def set_busy(self, busy: bool):
        self.btnRecompute.setEnabled(not busy)
        self.btnDefaults.setEnabled(not busy)
        self.btnReload.setEnabled(not busy)
        self.btnRecompute.setText("Computing..." if busy else "RECOMPUTE")
        QtWidgets.QApplication.processEvents()

    def clear_guard(self):
        self.lblGuard.setVisible(False)
        self.lblGuard.setText("")

    def set_guard(self, level: str, text: str):
        level = (level or "").lower().strip()
        if not text:
            self.clear_guard()
            return

        if level == "green":
            bg = "#e9f7ef"
            fg = "#1e6b3a"
            border = "#9ad0ad"
            prefix = "GREEN"
        elif level == "yellow":
            bg = "#fff7e6"
            fg = "#7a5200"
            border = "#ffd38a"
            prefix = "YELLOW"
        else:
            bg = "#fdecea"
            fg = "#7a1f1f"
            border = "#f5a7a0"
            prefix = "RED"

        self.lblGuard.setStyleSheet(
            f"background:{bg}; color:{fg}; border:1px solid {border}; "
            "padding:6px; border-radius:8px; font-size:12px;"
        )
        self.lblGuard.setText(f"[{prefix}] {text}")
        self.lblGuard.setVisible(True)

    def get_params(self) -> dict:
        p = dict(
            N=int(self.spinN.value()),
            BOUNDS=float(self.dblBounds.value()),
            ISO=float(self.dblIso.value()),
        )
        if self._plugin_get_params is not None:
            p.update(dict(self._plugin_get_params() or {}))
        return p

    def reset_defaults(self):
        self.spinN.setValue(220)
        self.dblBounds.setValue(1.50)
        self.dblIso.setValue(0.60)
        self.recompute_clicked.emit()

    def _on_plugin_combo_changed(self):
        pid = self.active_plugin_id()
        if pid:
            self.plugin_changed.emit(pid)
        # status is handled by main window status bar
