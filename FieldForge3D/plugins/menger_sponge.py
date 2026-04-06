# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/menger_sponge.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "menger_sponge", "name": "Menger Sponge (Nested Cubes)", "category": "Fibonacci"}


FORMULA = r"""Menger Sponge (nested cubes)
Plugin: menger_sponge  (menger_sponge.py)

Idea:
A 3D fractal built by repeatedly removing the center cross from a cube (Menger sponge).
We evaluate membership by iterating coordinates in base-3 style (or equivalent rules).

Rule of thumb:
At each level, if two or more coordinates fall into the middle third, the point is removed.

Output:
- v in [0..1] represents membership / proximity
- use GAIN to increase contrast when needed

Notes:
- LEVELS controls detail (and compute cost)
"""

def get_defaults() -> dict:
    return dict(
        DEPTH=4,
        THICK=0.08,
        GAIN=3.0,       # kontrast (0.5..10)
        ROT=0.0,
        INVERT=0,       # 0/1 (invert field)
    )

@njit(fastmath=True)
def rotz(x, y, a):
    ca = math.cos(a)
    sa = math.sin(a)
    return x*ca - y*sa, x*sa + y*ca

@njit(fastmath=True)
def menger_membership(x, y, z, depth):
    """Menger sponge implicit field."""
    ax = abs(x)
    ay = abs(y)
    az = abs(z)

    if ax > 1.0 or ay > 1.0 or az > 1.0:
        return 0

    u = (x + 1.0) * 0.5
    v = (y + 1.0) * 0.5
    w = (z + 1.0) * 0.5

    for _ in range(depth):
        # digit v base-3: 0,1,2
        du = int(u * 3.0)
        dv = int(v * 3.0)
        dw = int(w * 3.0)

        # “center third” = digit == 1
        cu = (du == 1)
        cv = (dv == 1)
        cw = (dw == 1)

        if (cu and cv) or (cu and cw) or (cv and cw):
            return 0

        u = u * 3.0 - du
        v = v * 3.0 - dv
        w = w * 3.0 - dw

    return 1

@njit(parallel=True, fastmath=True)
def menger_field(N, bounds, depth, thick, gain, rot, invert):
    N = int(N)
    bounds = float(bounds)
    depth = int(depth)
    thick = float(thick)
    gain = float(gain)
    rot = float(rot)
    invert = int(invert)

    field = np.zeros((N, N, N), dtype=np.float32)

    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    eps = max(1.0 / max(N, 1), 1e-4) * (1.0 / max(bounds, 1e-9))

    for ix in prange(N):
        x0 = float(xs[ix]) / max(bounds, 1e-9)
        for iy in range(N):
            y0 = float(ys[iy]) / max(bounds, 1e-9)
            for iz in range(N):
                z0 = float(zs[iz]) / max(bounds, 1e-9)

                x, y = rotz(x0, y0, rot)
                z = z0

                m0 = menger_membership(x, y, z, depth)

                if m0 == 0:
                    near = 0
                    if menger_membership(x + eps, y, z, depth): near += 1
                    if menger_membership(x - eps, y, z, depth): near += 1
                    if menger_membership(x, y + eps, z, depth): near += 1
                    if menger_membership(x, y - eps, z, depth): near += 1
                    if menger_membership(x, y, z + eps, depth): near += 1
                    if menger_membership(x, y, z - eps, depth): near += 1

                    # near=0..6 -> “soft shell”
                    acc = float(near) / 6.0
                    v = 1.0 - math.exp(-gain * acc)
                else:
                    out = 0
                    if menger_membership(x + eps, y, z, depth) == 0: out += 1
                    if menger_membership(x - eps, y, z, depth) == 0: out += 1
                    if menger_membership(x, y + eps, z, depth) == 0: out += 1
                    if menger_membership(x, y - eps, z, depth) == 0: out += 1
                    if menger_membership(x, y, z + eps, depth) == 0: out += 1
                    if menger_membership(x, y, z - eps, depth) == 0: out += 1

                    edge = float(out) / 6.0
                    acc = (1.0 - min(1.0, edge / max(thick, 1e-9)))
                    v = 1.0 - math.exp(-gain * acc)

                if invert != 0:
                    v = 1.0 - v

                if v < 0.0: v = 0.0
                if v > 1.0: v = 1.0
                field[ix, iy, iz] = np.float32(v)

    return field

def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    spD = QtWidgets.QSpinBox()
    spD.setRange(1, 6)
    spD.setValue(d["DEPTH"])

    spT = QtWidgets.QDoubleSpinBox()
    spT.setRange(0.01, 0.30)
    spT.setSingleStep(0.01)
    spT.setDecimals(2)
    spT.setValue(d["THICK"])

    spG = QtWidgets.QDoubleSpinBox()
    spG.setRange(0.2, 12.0)
    spG.setSingleStep(0.1)
    spG.setDecimals(2)
    spG.setValue(d["GAIN"])

    spR = QtWidgets.QDoubleSpinBox()
    spR.setRange(-6.28318, 6.28318)
    spR.setSingleStep(0.05)
    spR.setDecimals(2)
    spR.setValue(d["ROT"])

    spI = QtWidgets.QSpinBox()
    spI.setRange(0, 1)
    spI.setValue(d["INVERT"])

    f.addRow("DEPTH", spD)
    f.addRow("THICK", spT)
    f.addRow("GAIN", spG)
    f.addRow("ROT", spR)
    f.addRow("INVERT (0/1)", spI)

    def get_params():
        return dict(
            DEPTH=int(spD.value()),
            THICK=float(spT.value()),
            GAIN=float(spG.value()),
            ROT=float(spR.value()),
            INVERT=int(spI.value()),
        )

    return w, get_params

def compute(params: dict):
    return menger_field(
        params["N"], params["BOUNDS"],
        params["DEPTH"], params["THICK"], params["GAIN"],
        params["ROT"], params["INVERT"]
    )
