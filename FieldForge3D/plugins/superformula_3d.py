# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/superformula_3d.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "superformula_3d", "name": "Superformula 3D (Gielis)", "category": "Organic"}



FORMULA = r"""Superformula 3D (Gielis-style implicit surface)
Plugin: superformula_3d  (superformula_3d.py)

Superformula (2D)
  r(φ) = [ (|cos(mφ/4)/a|^n2 + |sin(mφ/4)/b|^n3) ]^(-1/n1)

3D extension
- Build two superformula radii: one for latitude θ and one for longitude φ.
- Combine them into a 3D surface.

Field output
- Produces a smooth, highly parameterized family of shapes:
  stars, flowers, spiky blobs, shells, etc.

Tips
- Small parameter changes can be dramatic.
- Use GAIN/ISO to control how "thick" the resulting surface looks.
"""

def get_defaults() -> dict:
    return dict(
        M1=7.0, M2=3.0,
        N1=0.25, N2=1.7, N3=1.7,
        A=1.0, B=1.0,
        THICK=0.10,
        GAIN=4.0,
        SCALE=1.0,
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
def super_r(angle: float, m: float, n1: float, n2: float, n3: float, a: float, b: float) -> float:
    # r = (|cos(m a/4)/a|^n2 + |sin(m a/4)/b|^n3)^(-1/n1)
    t1 = abs(math.cos(m * angle * 0.25) / max(a, 1e-9))
    t2 = abs(math.sin(m * angle * 0.25) / max(b, 1e-9))
    p = (t1 ** n2) + (t2 ** n3)
    if p <= 1e-12:
        return 0.0
    return p ** (-1.0 / max(n1, 1e-9))


@njit(parallel=True, fastmath=True)
def superformula_field(N, bounds, m1, m2, n1, n2, n3, a, b, thick, gain, scale):
    N = int(N)
    bounds = float(bounds)
    m1 = float(m1); m2 = float(m2)
    n1 = float(n1); n2 = float(n2); n3 = float(n3)
    a = float(a); b = float(b)
    thick = float(thick)
    gain = float(gain)
    scale = float(scale)

    field = np.zeros((N, N, N), dtype=np.float32)

    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    dx = (2.0 * bounds) / max(N - 1, 1)
    min_th = 1.0 * dx
    if thick < min_th:
        thick = min_th

    for i in prange(N):
        x0 = float(xs[i])
        for j in range(N):
            y0 = float(ys[j])
            for k in range(N):
                z0 = float(zs[k])

                x = x0 / max(scale, 1e-9)
                y = y0 / max(scale, 1e-9)
                z = z0 / max(scale, 1e-9)

                rho = math.sqrt(x * x + y * y + z * z) + 1e-12
                theta = math.atan2(y, x)  # -pi..pi
                phi = math.atan2(z, math.sqrt(x * x + y * y))  # -pi/2..pi/2

                r1 = super_r(theta, m1, n1, n2, n3, a, b)
                r2 = super_r(phi,   m2, n1, n2, n3, a, b)
                target = r1 * r2

                # implicit shell around rho = target
                d = abs(rho - target)
                t = 1.0 - (d / thick)
                t = clamp01(t)
                t = smoothstep01(t)

                out = 1.0 - math.exp(-gain * t)
                field[i, j, k] = np.float32(out)

    return field


def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    spM1 = QtWidgets.QDoubleSpinBox(); spM1.setRange(0.0, 20.0); spM1.setSingleStep(0.5); spM1.setDecimals(2); spM1.setValue(d["M1"])
    spM2 = QtWidgets.QDoubleSpinBox(); spM2.setRange(0.0, 20.0); spM2.setSingleStep(0.5); spM2.setDecimals(2); spM2.setValue(d["M2"])

    spN1 = QtWidgets.QDoubleSpinBox(); spN1.setRange(0.05, 8.0); spN1.setSingleStep(0.05); spN1.setDecimals(2); spN1.setValue(d["N1"])
    spN2 = QtWidgets.QDoubleSpinBox(); spN2.setRange(0.05, 8.0); spN2.setSingleStep(0.05); spN2.setDecimals(2); spN2.setValue(d["N2"])
    spN3 = QtWidgets.QDoubleSpinBox(); spN3.setRange(0.05, 8.0); spN3.setSingleStep(0.05); spN3.setDecimals(2); spN3.setValue(d["N3"])

    spA = QtWidgets.QDoubleSpinBox(); spA.setRange(0.2, 3.0); spA.setSingleStep(0.05); spA.setDecimals(2); spA.setValue(d["A"])
    spB = QtWidgets.QDoubleSpinBox(); spB.setRange(0.2, 3.0); spB.setSingleStep(0.05); spB.setDecimals(2); spB.setValue(d["B"])

    spS = QtWidgets.QDoubleSpinBox(); spS.setRange(0.3, 2.5); spS.setSingleStep(0.05); spS.setDecimals(2); spS.setValue(d["SCALE"])
    spT = QtWidgets.QDoubleSpinBox(); spT.setRange(0.01, 0.60); spT.setSingleStep(0.01); spT.setDecimals(2); spT.setValue(d["THICK"])
    spG = QtWidgets.QDoubleSpinBox(); spG.setRange(0.2, 10.0); spG.setSingleStep(0.1); spG.setDecimals(2); spG.setValue(d["GAIN"])

    f.addRow("M1", spM1)
    f.addRow("M2", spM2)
    f.addRow("N1", spN1)
    f.addRow("N2", spN2)
    f.addRow("N3", spN3)
    f.addRow("A", spA)
    f.addRow("B", spB)
    f.addRow("SCALE", spS)
    f.addRow("THICK", spT)
    f.addRow("GAIN", spG)

    def get_params():
        return dict(
            M1=float(spM1.value()),
            M2=float(spM2.value()),
            N1=float(spN1.value()),
            N2=float(spN2.value()),
            N3=float(spN3.value()),
            A=float(spA.value()),
            B=float(spB.value()),
            SCALE=float(spS.value()),
            THICK=float(spT.value()),
            GAIN=float(spG.value()),
        )

    return w, get_params


def compute(params: dict):
    return superformula_field(
        params["N"], params["BOUNDS"],
        params["M1"], params["M2"],
        params["N1"], params["N2"], params["N3"],
        params["A"], params["B"],
        params["THICK"], params["GAIN"],
        params["SCALE"],
    )
