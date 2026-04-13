# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/golden_lattice.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "golden_lattice", "name": "Golden Lattice (φ)", "category": "Math Art"}


FORMULA = r"""Golden lattice (φ-mix)
Plugin: golden_lattice  (golden_lattice.py)

Idea:
Two periodic wave-fields are phase-shifted by the golden ratio φ and blended into one
interference lattice. The result looks like a "woven" crystal with φ-flavored symmetry.

Formulas:
- φ = (1 + sqrt(5)) / 2
- a = sin(φx + y) * cos(φy + z)
- b = sin(φz + x) * cos(φx + z)
- g = mix * a + (1 - mix) * b
- v = 0.5 + 0.5 * g   (optionally clamped to [0..1])

Notes:
- freq scales space: (x, y, z) := freq * (x, y, z)
- mix blends the two φ-shifted terms into a single interference field
- clamp keeps values inside [0..1] for stable isosurfaces
"""

def get_defaults() -> dict:
    return dict(
        FREQ=5.0,
        MIX=0.65,   # mix sin/cos
        CLAMP=True
    )

@njit(parallel=True, fastmath=True)
def golden_field(N, bounds, freq, mix, clamp_flag):
    N=int(N); bounds=float(bounds)
    freq=float(freq); mix=float(mix)
    phi = (1.0 + math.sqrt(5.0)) * 0.5

    field = np.zeros((N,N,N), dtype=np.float32)
    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    for ix in prange(N):
        x = float(xs[ix]) * freq
        for iy in range(N):
            y = float(ys[iy]) * freq
            for iz in range(N):
                z = float(zs[iz]) * freq

                a = math.sin(phi*x + y) * math.cos(phi*y + z)
                b = math.sin(phi*z + x) * math.cos(phi*x + z)
                g = mix * a + (1.0 - mix) * b

                v = 0.5 + 0.5 * g
                if clamp_flag == 1:
                    if v < 0.0: v = 0.0
                    if v > 1.0: v = 1.0
                field[ix,iy,iz] = np.float32(v)

    return field

def build_ui(parent):
    d=get_defaults()
    w=QtWidgets.QWidget(parent)
    f=QtWidgets.QFormLayout(w)

    spF = QtWidgets.QDoubleSpinBox(); spF.setRange(0.5, 20.0); spF.setSingleStep(0.5); spF.setDecimals(2); spF.setValue(d["FREQ"])
    spM = QtWidgets.QDoubleSpinBox(); spM.setRange(0.0, 1.0); spM.setSingleStep(0.05); spM.setDecimals(2); spM.setValue(d["MIX"])
    chk = QtWidgets.QCheckBox("Clamp [0..1]"); chk.setChecked(bool(d["CLAMP"]))

    f.addRow("FREQ", spF)
    f.addRow("MIX", spM)
    f.addRow("", chk)

    def get_params():
        return dict(FREQ=float(spF.value()), MIX=float(spM.value()), CLAMP=bool(chk.isChecked()))
    return w, get_params

def compute(params: dict):
    clamp_flag = 1 if bool(params.get("CLAMP", True)) else 0
    return golden_field(params["N"], params["BOUNDS"], params["FREQ"], params["MIX"], clamp_flag)
