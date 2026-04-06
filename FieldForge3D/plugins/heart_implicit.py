# FieldForge 3D (FieldForge3D)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file

# plugins/heart_implicit.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "heart_implicit", "name": "Heart (implicit)", "category": "Field"}

FORMULA = r"""Heart (implicit 3D)
Classic implicit heart surface:
  F(x,y,z) = (x^2 + 9/4 y^2 + z^2 - 1)^3 - x^2 z^3 - 9/80 y^2 z^3

We map F to [0..1] around the surface with thickness/softness.
"""

def get_defaults() -> dict:
    return dict(
        SCALE=0.85,   # overall size inside bounds
        THICK=0.12,   # shell thickness
        SOFT=0.20,    # softness
        FLIP_Z=True,  # nicer orientation
    )

@njit(parallel=True, fastmath=True)
def heart_field(N, bounds, scale, thick, soft, flip_z: int):
    N = int(N)
    bounds = float(bounds)
    scale = float(scale)
    thick = float(thick)
    soft = float(soft)

    field = np.zeros((N, N, N), dtype=np.float32)

    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    inv_soft = 1.0 / max(1e-6, soft)

    for ix in prange(N):
        x = float(xs[ix]) / scale
        x2 = x*x
        for iy in range(N):
            y = float(ys[iy]) / scale
            y2 = y*y
            for iz in range(N):
                z = float(zs[iz]) / scale
                if flip_z == 1:
                    z = -z

                z2 = z*z
                a = x2 + 2.25*y2 + z2 - 1.0
                F = a*a*a - x2*z*z2 - (9.0/80.0)*y2*z*z2

                # distance-ish: use |F|, not exact SDF but good visually
                d = abs(F)

                # shell around F=0:
                a2 = (d - thick) * inv_soft
                if a2 <= -1.0:
                    v = 1.0
                elif a2 >= 1.0:
                    v = 0.0
                else:
                    u = 0.5 * (1.0 - a2)
                    v = u*u*(3.0 - 2.0*u)

                field[ix, iy, iz] = np.float32(v)

    return field

def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    sp_scale = QtWidgets.QDoubleSpinBox()
    sp_scale.setRange(0.30, 2.00)
    sp_scale.setDecimals(2)
    sp_scale.setSingleStep(0.05)
    sp_scale.setValue(d["SCALE"])

    sp_thick = QtWidgets.QDoubleSpinBox()
    sp_thick.setRange(0.01, 0.60)
    sp_thick.setDecimals(3)
    sp_thick.setSingleStep(0.01)
    sp_thick.setValue(d["THICK"])

    sp_soft = QtWidgets.QDoubleSpinBox()
    sp_soft.setRange(0.01, 0.60)
    sp_soft.setDecimals(3)
    sp_soft.setSingleStep(0.01)
    sp_soft.setValue(d["SOFT"])

    chk_flip = QtWidgets.QCheckBox("FLIP_Z (nicer)")
    chk_flip.setChecked(bool(d["FLIP_Z"]))

    f.addRow("SCALE", sp_scale)
    f.addRow("THICK", sp_thick)
    f.addRow("SOFT", sp_soft)
    f.addRow("", chk_flip)

    def get_params():
        return dict(
            SCALE=float(sp_scale.value()),
            THICK=float(sp_thick.value()),
            SOFT=float(sp_soft.value()),
            FLIP_Z=bool(chk_flip.isChecked()),
        )

    return w, get_params

def compute(params: dict):
    return heart_field(
        params["N"],
        params["BOUNDS"],
        params.get("SCALE", 0.85),
        params.get("THICK", 0.12),
        params.get("SOFT", 0.20),
        1 if bool(params.get("FLIP_Z", True)) else 0,
    )
