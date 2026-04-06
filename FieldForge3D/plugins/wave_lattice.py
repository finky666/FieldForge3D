# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/wave_lattice.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "wave_lattice", "name": "Wave Lattice", "category": "Math Art"}


FORMULA = r"""Wave lattice (sin + cos "turbulence")
Plugin: wave_lattice  (wave_lattice.py)

Idea:
A smooth lattice made from three coupled trigonometric waves. The TWIST term adds a mild
cross-axis modulation (turbulence), producing organic "folds" while staying periodic.

Formulas:
- v0 = sin(x + TWIST*cos(y))
     + sin(y + TWIST*cos(z))
     + sin(z + TWIST*cos(x))
- v  = clamp( 0.5 + 0.5 * tanh( GAIN * v0 / 3 ) )

Notes:
- freq scales space: (x, y, z) := freq * (x, y, z)
- higher GAIN increases contrast (sharper walls)
"""

def get_defaults() -> dict:
    return dict(
        FREQ=6.0,
        TWIST=1.2,
        GAIN=1.0
    )

@njit(parallel=True, fastmath=True)
def wave_field(N, bounds, freq, twist, gain):
    N=int(N); bounds=float(bounds)
    freq=float(freq); twist=float(twist); gain=float(gain)

    field=np.zeros((N,N,N), dtype=np.float32)
    xs=np.linspace(-bounds,bounds,N).astype(np.float32)
    ys=np.linspace(-bounds,bounds,N).astype(np.float32)
    zs=np.linspace(-bounds,bounds,N).astype(np.float32)

    for ix in prange(N):
        x=float(xs[ix]) * freq
        for iy in range(N):
            y=float(ys[iy]) * freq
            for iz in range(N):
                z=float(zs[iz]) * freq

                # pseudo turbulence
                v = math.sin(x + twist*math.cos(y)) \
                    + math.sin(y + twist*math.cos(z)) \
                    + math.sin(z + twist*math.cos(x))

                v = 0.5 + 0.5 * math.tanh(gain * v / 3.0)
                if v < 0.0: v = 0.0
                if v > 1.0: v = 1.0
                field[ix,iy,iz]=np.float32(v)

    return field

def build_ui(parent):
    d=get_defaults()
    w=QtWidgets.QWidget(parent)
    f=QtWidgets.QFormLayout(w)

    spF=QtWidgets.QDoubleSpinBox(); spF.setRange(0.5, 25.0); spF.setSingleStep(0.5); spF.setDecimals(2); spF.setValue(d["FREQ"])
    spT=QtWidgets.QDoubleSpinBox(); spT.setRange(0.0, 5.0); spT.setSingleStep(0.1); spT.setDecimals(2); spT.setValue(d["TWIST"])
    spG=QtWidgets.QDoubleSpinBox(); spG.setRange(0.1, 5.0); spG.setSingleStep(0.1); spG.setDecimals(2); spG.setValue(d["GAIN"])

    f.addRow("FREQ", spF)
    f.addRow("TWIST", spT)
    f.addRow("GAIN", spG)

    def get_params():
        return dict(FREQ=float(spF.value()), TWIST=float(spT.value()), GAIN=float(spG.value()))
    return w, get_params

def compute(params: dict):
    return wave_field(params["N"], params["BOUNDS"], params["FREQ"], params["TWIST"], params["GAIN"])
