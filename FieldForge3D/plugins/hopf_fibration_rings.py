# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/hopf_fibration_rings.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "hopf_fibration_rings", "name": "Hopf-ish Fibration Rings", "category": "Topology"}



FORMULA = r"""Hopf Fibration Rings (field)
Plugin: hopf_fibration_rings  (hopf_fibration_rings.py)

Idea
- A stylized 3D projection inspired by the Hopf fibration:
  linked circles / tori-like rings that "thread" through each other.

Implementation sketch
- We build a field from distances to a set of rings.
- Each ring contributes a smooth radial profile.
- Multiple rings are combined to form a linked structure.

Output mapping
  value = 1 - exp(-GAIN * field)

Tips
- Increase RINGS for more links.
- If it becomes too dense: reduce THICK or reduce GAIN.
"""

def get_defaults() -> dict:
    return dict(
        R=0.55,
        r=0.11,
        COUNT=8,
        TWIST=1.2,
        THICK=0.10,
        GAIN=4.0,       # kontrast
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
def torus_sdf(x: float, y: float, z: float, R: float, r: float) -> float:
    qx = math.sqrt(x * x + y * y) - R
    return math.sqrt(qx * qx + z * z) - r


@njit(parallel=True, fastmath=True)
def hopf_rings_field(N, bounds, Rrel, rrel, count, twist, thick, gain):
    N = int(N)
    bounds = float(bounds)
    Rrel = float(Rrel)
    rrel = float(rrel)
    count = int(count)
    twist = float(twist)
    thick = float(thick)
    gain = float(gain)

    field = np.zeros((N, N, N), dtype=np.float32)

    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    dx = (2.0 * bounds) / max(N - 1, 1)
    min_th = 1.0 * dx
    if thick < min_th:
        thick = min_th

    R = Rrel * bounds
    r = rrel * bounds

    for i in prange(N):
        x0 = float(xs[i])
        for j in range(N):
            y0 = float(ys[j])
            for k in range(N):
                z0 = float(zs[k])

                best = 0.0

                for n in range(count):
                    t = (2.0 * math.pi * n) / max(count, 1)
                    ax = twist * 0.30 * n
                    ay = twist * 0.22 * n
                    az = t + twist * 0.18 * n

                    x, y, z = rot3(x0, y0, z0, ax, ay, az)

                    # striedanie rovin ringov
                    m = n % 3
                    if m == 0:
                        d = torus_sdf(x, y, z, R, r)
                    elif m == 1:
                        d = torus_sdf(x, z, y, R, r)
                    else:
                        d = torus_sdf(y, z, x, R, r)

                    ad = abs(d)
                    s = 1.0 - (ad / thick)
                    s = clamp01(s)
                    s = smoothstep01(s)

                    if s > best:
                        best = s

                out = 1.0 - math.exp(-gain * best)
                field[i, j, k] = np.float32(out)

    return field


def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    spR = QtWidgets.QDoubleSpinBox(); spR.setRange(0.10, 1.20); spR.setSingleStep(0.01); spR.setDecimals(2); spR.setValue(d["R"])
    spr = QtWidgets.QDoubleSpinBox(); spr.setRange(0.02, 0.40); spr.setSingleStep(0.01); spr.setDecimals(2); spr.setValue(d["r"])
    spC = QtWidgets.QSpinBox(); spC.setRange(4, 16); spC.setValue(int(d["COUNT"]))
    spTw = QtWidgets.QDoubleSpinBox(); spTw.setRange(0.0, 4.0); spTw.setSingleStep(0.05); spTw.setDecimals(2); spTw.setValue(d["TWIST"])
    spT = QtWidgets.QDoubleSpinBox(); spT.setRange(0.01, 0.60); spT.setSingleStep(0.01); spT.setDecimals(2); spT.setValue(d["THICK"])
    spG = QtWidgets.QDoubleSpinBox(); spG.setRange(0.2, 10.0); spG.setSingleStep(0.1); spG.setDecimals(2); spG.setValue(d["GAIN"])

    f.addRow("R", spR)
    f.addRow("r", spr)
    f.addRow("COUNT", spC)
    f.addRow("TWIST", spTw)
    f.addRow("THICK", spT)
    f.addRow("GAIN", spG)

    def get_params():
        return dict(
            R=float(spR.value()),
            r=float(spr.value()),
            COUNT=int(spC.value()),
            TWIST=float(spTw.value()),
            THICK=float(spT.value()),
            GAIN=float(spG.value()),
        )

    return w, get_params


def compute(params: dict):
    return hopf_rings_field(
        params["N"], params["BOUNDS"],
        params["R"], params["r"],
        params["COUNT"], params["TWIST"],
        params["THICK"], params["GAIN"],
    )
