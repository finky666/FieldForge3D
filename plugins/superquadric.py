# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/superquadric.py
from __future__ import annotations

import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "superquadric", "name": "Superquadric (implicit)", "category": "Shapes"}


FORMULA = r"""Superquadric (implicit)
Plugin: superquadric  (superquadric.py)

Idea:
An implicit family of shapes between cubes and spheres, controlled by exponents.

Formula:
- (|x/a|^n + |y/b|^n)^(m/n) + |z/c|^m = 1

Notes:
- larger exponents -> more boxy
- smaller exponents -> more round / star-like
"""

def get_defaults() -> dict:
    return dict(
        A=1.0, B=1.0, C=1.0,
        E1=0.50,               # exponent 1
        E2=0.50,               # exponent 2
        SOFT=0.25
    )

@njit(parallel=True, fastmath=True)
def superquadric_field(N, bounds, A, B, C, e1, e2, soft):
    N = int(N); bounds=float(bounds)
    A=float(A); B=float(B); C=float(C)
    e1=float(e1); e2=float(e2); soft=float(soft)

    field = np.zeros((N,N,N), dtype=np.float32)
    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    eps = 1e-12
    for ix in prange(N):
        x = float(xs[ix]) / max(A, eps)
        ax = abs(x)
        for iy in range(N):
            y = float(ys[iy]) / max(B, eps)
            ay = abs(y)
            for iz in range(N):
                z = float(zs[iz]) / max(C, eps)
                az = abs(z)

                # implicit superquadric:
                # ( (|x|^(2/e2) + |y|^(2/e2))^(e2/e1) + |z|^(2/e1) ) = 1
                t = (ax**(2.0/max(e2,eps)) + ay**(2.0/max(e2,eps)))
                t = t**(max(e2,eps)/max(e1,eps))
                val = t + az**(2.0/max(e1,eps)) - 1.0  # 0 surface

                v = 0.5 - val / max(soft, 1e-6)
                if v < 0.0: v = 0.0
                if v > 1.0: v = 1.0
                field[ix,iy,iz] = np.float32(v)

    return field

def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    spA = QtWidgets.QDoubleSpinBox(); spA.setRange(0.2, 3.0); spA.setSingleStep(0.1); spA.setDecimals(2); spA.setValue(d["A"])
    spB = QtWidgets.QDoubleSpinBox(); spB.setRange(0.2, 3.0); spB.setSingleStep(0.1); spB.setDecimals(2); spB.setValue(d["B"])
    spC = QtWidgets.QDoubleSpinBox(); spC.setRange(0.2, 3.0); spC.setSingleStep(0.1); spC.setDecimals(2); spC.setValue(d["C"])

    spE1 = QtWidgets.QDoubleSpinBox(); spE1.setRange(0.10, 2.0); spE1.setSingleStep(0.05); spE1.setDecimals(2); spE1.setValue(d["E1"])
    spE2 = QtWidgets.QDoubleSpinBox(); spE2.setRange(0.10, 2.0); spE2.setSingleStep(0.05); spE2.setDecimals(2); spE2.setValue(d["E2"])

    spS = QtWidgets.QDoubleSpinBox(); spS.setRange(0.02, 1.0); spS.setSingleStep(0.02); spS.setDecimals(2); spS.setValue(d["SOFT"])

    f.addRow("A", spA); f.addRow("B", spB); f.addRow("C", spC)
    f.addRow("E1", spE1); f.addRow("E2", spE2)
    f.addRow("SOFT", spS)

    def get_params():
        return dict(A=float(spA.value()), B=float(spB.value()), C=float(spC.value()),
                    E1=float(spE1.value()), E2=float(spE2.value()), SOFT=float(spS.value()))
    return w, get_params

def compute(params: dict):
    return superquadric_field(params["N"], params["BOUNDS"], params["A"], params["B"], params["C"],
                              params["E1"], params["E2"], params["SOFT"])
