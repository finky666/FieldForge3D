# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/mandelbox_like.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "mandelbox_like", "name": "Mandelbox-like (escape)", "category": "Fractals"}


FORMULA = r"""Mandelbox-like Fractal (distance-ish field)
Plugin: mandelbox_like  (mandelbox_like.py)

Idea
- A "Mandelbox-inspired" iterative fold + scale process in 3D.
- Not a strict distance estimator, but produces characteristic boxy fractal structures.

Typical steps
- Box fold: reflect coordinates into a cube.
- Sphere fold: clamp radius into a range.
- Scale + translate: p = p * SCALE + offset

Field
- The iteration produces a value that correlates with "escape / detail".
- Mapped through an exponential curve for a usable [0..1] scalar field.

Tips
- Higher ITER increases detail but costs time.
- Adjust SCALE/SHIFT to explore different shapes.
"""

def get_defaults() -> dict:
    return dict(
        SCALE=-1.8,
        MIN_RAD=0.5,
        FIX_RAD=1.0,
        ITERS=18,
        BAIL=4.0,
    )

@njit(fastmath=True)
def box_fold(a, fold=1.0):
    if a > fold:  return 2.0*fold - a
    if a < -fold: return -2.0*fold - a
    return a

@njit(parallel=True, fastmath=True)
def mandelbox_escape_field(N, bounds, scale, min_rad, fix_rad, iters, bail):
    N=int(N); bounds=float(bounds)
    scale=float(scale); min_rad=float(min_rad); fix_rad=float(fix_rad)
    iters=int(iters); bail=float(bail)

    field=np.zeros((N,N,N), dtype=np.float32)
    xs=np.linspace(-bounds,bounds,N).astype(np.float32)
    ys=np.linspace(-bounds,bounds,N).astype(np.float32)
    zs=np.linspace(-bounds,bounds,N).astype(np.float32)

    bail2 = bail*bail
    eps = 1e-12

    for ix in prange(N):
        x0=float(xs[ix])
        for iy in range(N):
            y0=float(ys[iy])
            for iz in range(N):
                z0=float(zs[iz])

                x=x0; y=y0; z=z0
                k = 0
                for t in range(iters):
                    # box fold
                    x = box_fold(x)
                    y = box_fold(y)
                    z = box_fold(z)

                    # sphere fold
                    r2 = x*x + y*y + z*z + eps
                    if r2 < min_rad*min_rad:
                        m = (fix_rad*fix_rad)/(min_rad*min_rad)
                        x *= m; y *= m; z *= m
                    elif r2 < fix_rad*fix_rad:
                        m = (fix_rad*fix_rad)/r2
                        x *= m; y *= m; z *= m

                    # scale + translate
                    x = x*scale + x0
                    y = y*scale + y0
                    z = z*scale + z0

                    if (x*x + y*y + z*z) > bail2:
                        break
                    k = t+1

                # escape -> [0..1]
                v = 1.0 if k >= iters else (k / iters)
                field[ix,iy,iz] = np.float32(v)
    return field

def build_ui(parent):
    d=get_defaults()
    w=QtWidgets.QWidget(parent)
    f=QtWidgets.QFormLayout(w)

    spS=QtWidgets.QDoubleSpinBox(); spS.setRange(-3.0, 3.0); spS.setSingleStep(0.1); spS.setDecimals(2); spS.setValue(d["SCALE"])
    spMin=QtWidgets.QDoubleSpinBox(); spMin.setRange(0.1, 2.0); spMin.setSingleStep(0.05); spMin.setDecimals(2); spMin.setValue(d["MIN_RAD"])
    spFix=QtWidgets.QDoubleSpinBox(); spFix.setRange(0.2, 3.0); spFix.setSingleStep(0.05); spFix.setDecimals(2); spFix.setValue(d["FIX_RAD"])
    spI=QtWidgets.QSpinBox(); spI.setRange(5, 60); spI.setValue(d["ITERS"])
    spB=QtWidgets.QDoubleSpinBox(); spB.setRange(1.5, 10.0); spB.setSingleStep(0.5); spB.setDecimals(2); spB.setValue(d["BAIL"])

    f.addRow("SCALE", spS)
    f.addRow("MIN_RAD", spMin)
    f.addRow("FIX_RAD", spFix)
    f.addRow("ITERS", spI)
    f.addRow("BAIL", spB)

    def get_params():
        return dict(SCALE=float(spS.value()), MIN_RAD=float(spMin.value()), FIX_RAD=float(spFix.value()),
                    ITERS=int(spI.value()), BAIL=float(spB.value()))
    return w, get_params

def compute(params: dict):
    return mandelbox_escape_field(params["N"], params["BOUNDS"],
                                  params["SCALE"], params["MIN_RAD"], params["FIX_RAD"],
                                  params["ITERS"], params["BAIL"])
