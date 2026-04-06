# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/torus.py
from __future__ import annotations

import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "torus", "name": "Torus (implicit)", "category": "Shapes"}


FORMULA = r"""Torus (SDF-like implicit)
Plugin: torus  (torus.py)

Idea:
Classic torus defined by major radius R and minor radius r.

Implicit / SDF-style:
- q = (sqrt(x^2 + y^2) - R, z)
- d = sqrt(qx^2 + qy^2) - r
- we convert distance to a smooth field around d=0

Notes:
- ISO controls the shell thickness around the torus surface
"""

def get_defaults() -> dict:
    return dict(
        R=0.75,
        r=0.28,   # polomer trubky
        SOFT=0.20
    )

@njit(parallel=True, fastmath=True)
def torus_field(N, bounds, R, r, soft):
    N = int(N)
    bounds = float(bounds)
    R = float(R)
    r = float(r)
    soft = float(soft)

    field = np.zeros((N, N, N), dtype=np.float32)
    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    for ix in prange(N):
        x = float(xs[ix])
        for iy in range(N):
            y = float(ys[iy])
            q = (x*x + y*y) ** 0.5
            for iz in range(N):
                z = float(zs[iz])

                # implicit: (sqrt(x^2+y^2)-R)^2 + z^2 = r^2
                d = (q - R)*(q - R) + z*z - r*r  # 0 = surface

                # map to [0..1], inside-ish -> 1, outside -> 0 (soft step)
                v = 0.5 - d / max(soft, 1e-6)
                if v < 0.0: v = 0.0
                if v > 1.0: v = 1.0
                field[ix, iy, iz] = np.float32(v)

    return field

def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    spR = QtWidgets.QDoubleSpinBox(); spR.setRange(0.1, 2.0); spR.setSingleStep(0.05); spR.setDecimals(2); spR.setValue(d["R"])
    spr = QtWidgets.QDoubleSpinBox(); spr.setRange(0.05, 1.0); spr.setSingleStep(0.02); spr.setDecimals(2); spr.setValue(d["r"])
    spS = QtWidgets.QDoubleSpinBox(); spS.setRange(0.02, 1.0); spS.setSingleStep(0.02); spS.setDecimals(2); spS.setValue(d["SOFT"])

    f.addRow("R (main)", spR)
    f.addRow("r (tube)", spr)
    f.addRow("SOFT", spS)

    def get_params():
        return dict(R=float(spR.value()), r=float(spr.value()), SOFT=float(spS.value()))
    return w, get_params

def compute(params: dict):
    return torus_field(params["N"], params["BOUNDS"], params["R"], params["r"], params["SOFT"])
