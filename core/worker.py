# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# core/worker.py
from __future__ import annotations

import time
from PyQt6 import QtCore


class ComputeWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, dict, float)  # field, params, seconds
    failed = QtCore.pyqtSignal(str)

    def __init__(self, plugin_module, params: dict):
        super().__init__()
        self.plugin_module = plugin_module
        self.params = dict(params)

    @QtCore.pyqtSlot()
    def run(self):
        try:
            t0 = time.time()
            field = self.plugin_module.compute(self.params)
            t1 = time.time()
            self.finished.emit(field, self.params, float(t1 - t0))
        except Exception as e:
            self.failed.emit(repr(e))
