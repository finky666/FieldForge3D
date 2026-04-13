# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/gyroid.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets


PLUGIN_META = {"id": "gyroid", "name": "Gyroid (TPMS)", "category": "Field"}



FORMULA = r"""Gyroid (TPMS)
Plugin: gyroid  (gyroid.py)

Idea:
The classic triply-periodic minimal surface (TPMS) "gyroid". The isosurface of F(x,y,z)=0
creates a continuous labyrinth-like sheet.

Formula:
- F(x,y,z) = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)
- optional "abs/sym" modes may fold / symmetrize the field
- output mapping typically uses v = clamp(0.5 + 0.5 * F)

Notes:
- freq scales space: (x, y, z) := freq * (x, y, z)
- ISO selects the shell thickness around F=0
"""

def get_defaults() -> dict:
    return dict(
        FREQ=3.0,      # frequency
        MODE="abs",    # abs | raw
    )


@njit(parallel=True, fastmath=True)
def gyroid_field(N, bounds, freq, mode_abs: int):
    bounds = float(bounds)
    freq = float(freq)
    N = int(N)

    field = np.zeros((N, N, N), dtype=np.float32)

    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    # Gyroid: g = sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)  ~ [-1.5..1.5]
    # map to [0..1]
    inv = np.float32(1.0 / 3.0)
    shift = np.float32(1.5)

    for ix in prange(N):
        x = float(xs[ix]) * freq
        sx = math.sin(x)
        cx = math.cos(x)
        for iy in range(N):
            y = float(ys[iy]) * freq
            sy = math.sin(y)
            cy = math.cos(y)
            for iz in range(N):
                z = float(zs[iz]) * freq
                sz = math.sin(z)
                cz = math.cos(z)

                g = sx * cy + sy * cz + sz * cx
                if mode_abs == 1:
                    g = abs(g)

                v = (g + shift) * inv
                if v < 0.0:
                    v = 0.0
                elif v > 1.0:
                    v = 1.0
                field[ix, iy, iz] = np.float32(v)

    return field


def build_ui(parent):
    d = get_defaults()

    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    sp_freq = QtWidgets.QDoubleSpinBox()
    sp_freq.setRange(0.25, 12.0)
    sp_freq.setSingleStep(0.25)
    sp_freq.setDecimals(2)
    sp_freq.setValue(d["FREQ"])

    cmb_mode = QtWidgets.QComboBox()
    cmb_mode.addItem("abs (sym)", "abs")
    cmb_mode.addItem("raw", "raw")
    cmb_mode.setCurrentIndex(0 if d["MODE"] == "abs" else 1)

    f.addRow("FREQ", sp_freq)
    f.addRow("MODE", cmb_mode)

    def get_params():
        return dict(
            FREQ=float(sp_freq.value()),
            MODE=str(cmb_mode.currentData()),
        )

    return w, get_params


def compute(params: dict):
    mode_abs = 1 if str(params.get("MODE", "abs")) == "abs" else 0
    return gyroid_field(params["N"], params["BOUNDS"], params["FREQ"], mode_abs)
