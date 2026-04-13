# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/klein_bottle_field.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "klein_bottle_field", "name": "Klein Bottle (implicit)", "category": "Topology"}



FORMULA = r"""Klein bottle (implicit, shell)
Plugin: klein_bottle_field  (klein_bottle_field.py)

Formulas:
  - r² = x² + y² + z²
  - F(x,y,z) = (r² + 2y - 1)² · (r² - 2y - 1) - 8z²·(r² + 2y - 1)

  - shell around F = 0:
      t = smoothstep( clamp(1 - |F| / THICK) )
      v = 1 - exp(-GAIN · t)

Notes:
"""

def get_defaults() -> dict:
    return dict(
        A=2.0,
        THICK=0.08,
        GAIN=4.0,       # contrast
        ROTX=0.0,
        ROTY=0.0,
        ROTZ=0.0,
    )


@njit(fastmath=True)
def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@njit(fastmath=True)
def smoothstep01(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


@njit(fastmath=True)
def rot3(x: float, y: float, z: float, ax: float, ay: float, az: float):
    cx = math.cos(ax); sx = math.sin(ax)
    y1 = cx * y - sx * z
    z1 = sx * y + cx * z

    cy = math.cos(ay); sy = math.sin(ay)
    x2 = cy * x + sy * z1
    z2 = -sy * x + cy * z1

    cz = math.cos(az); sz = math.sin(az)
    x3 = cz * x2 - sz * y1
    y3 = sz * x2 + cz * y1

    return x3, y3, z2


@njit(fastmath=True)
def klein_implicit(x: float, y: float, z: float, a: float) -> float:
    """Implicit Klein bottle field."""
    x = x / max(a, 1e-9)
    y = y / max(a, 1e-9)
    z = z / max(a, 1e-9)

    x2 = x * x
    y2 = y * y
    z2 = z * z
    r2 = x2 + y2 + z2

    # F = (r2 + 2y - 1)^2 * (r2 - 2y - 1) - 8 z^2 (r2 + 2y - 1)
    t = r2 + 2.0 * y - 1.0
    F = (t * t) * (r2 - 2.0 * y - 1.0) - 8.0 * z2 * t
    return F


@njit(parallel=True, fastmath=True)
def klein_field(N, bounds, a, thick, gain, rx, ry, rz):
    N = int(N)
    bounds = float(bounds)
    a = float(a)
    thick = float(thick)
    gain = float(gain)
    rx = float(rx); ry = float(ry); rz = float(rz)

    field = np.zeros((N, N, N), dtype=np.float32)

    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    dx = (2.0 * bounds) / max(N - 1, 1)
    min_th = 1.2 * dx
    if thick < min_th:
        thick = min_th

    for i in prange(N):
        x0 = float(xs[i])
        for j in range(N):
            y0 = float(ys[j])
            for k in range(N):
                z0 = float(zs[k])

                x, y, z = rot3(x0, y0, z0, rx, ry, rz)
                F = klein_implicit(x, y, z, a)

                # shell around F=0
                ad = abs(F)
                t = 1.0 - (ad / thick)
                t = clamp01(t)
                t = smoothstep01(t)

                out = 1.0 - math.exp(-gain * t)
                field[i, j, k] = np.float32(out)

    return field


def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    spA = QtWidgets.QDoubleSpinBox(); spA.setRange(0.5, 6.0); spA.setSingleStep(0.1); spA.setDecimals(2); spA.setValue(d["A"])
    spT = QtWidgets.QDoubleSpinBox(); spT.setRange(0.01, 0.50); spT.setSingleStep(0.01); spT.setDecimals(2); spT.setValue(d["THICK"])
    spG = QtWidgets.QDoubleSpinBox(); spG.setRange(0.2, 10.0); spG.setSingleStep(0.1); spG.setDecimals(2); spG.setValue(d["GAIN"])

    spRX = QtWidgets.QDoubleSpinBox(); spRX.setRange(-3.14, 3.14); spRX.setSingleStep(0.05); spRX.setDecimals(2); spRX.setValue(d["ROTX"])
    spRY = QtWidgets.QDoubleSpinBox(); spRY.setRange(-3.14, 3.14); spRY.setSingleStep(0.05); spRY.setDecimals(2); spRY.setValue(d["ROTY"])
    spRZ = QtWidgets.QDoubleSpinBox(); spRZ.setRange(-3.14, 3.14); spRZ.setSingleStep(0.05); spRZ.setDecimals(2); spRZ.setValue(d["ROTZ"])

    f.addRow("A", spA)
    f.addRow("THICK", spT)
    f.addRow("GAIN", spG)
    f.addRow("ROTX", spRX)
    f.addRow("ROTY", spRY)
    f.addRow("ROTZ", spRZ)

    def get_params():
        return dict(
            A=float(spA.value()),
            THICK=float(spT.value()),
            GAIN=float(spG.value()),
            ROTX=float(spRX.value()),
            ROTY=float(spRY.value()),
            ROTZ=float(spRZ.value()),
        )

    return w, get_params


def compute(params: dict):
    return klein_field(
        params["N"], params["BOUNDS"],
        params["A"], params["THICK"], params["GAIN"],
        params["ROTX"], params["ROTY"], params["ROTZ"],
    )
