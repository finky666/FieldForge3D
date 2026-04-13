# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "tpms_schwarz_p","name":"TPMS Schwarz-P (periodic)","category":"Surfaces"}


FORMULA = r"""TPMS Schwarz P (variant)
Plugin: tpms_schwarz_p  (tpms_schwarz_p.py)

Idea:
A Schwarz P TPMS variant used as an isosurface-based lattice.

Formula:
- F = cos(x) + cos(y) + cos(z)   (with optional warps / modes)

Notes:
- freq scales space
- ISO selects the shell thickness around F=0
"""

def get_defaults():
    return dict(FREQ=3.0, GAIN=2.0)

@njit(parallel=True, fastmath=True)
def schwarz_p_field(N, bounds, freq, gain):
    N=int(N); bounds=float(bounds); freq=float(freq); gain=float(gain)
    field=np.zeros((N,N,N), dtype=np.float32)
    xs=np.linspace(-bounds,bounds,N).astype(np.float32)
    ys=np.linspace(-bounds,bounds,N).astype(np.float32)
    zs=np.linspace(-bounds,bounds,N).astype(np.float32)

    for ix in prange(N):
        x=float(xs[ix])*freq
        cx=math.cos(x)
        for iy in range(N):
            y=float(ys[iy])*freq
            cy=math.cos(y)
            for iz in range(N):
                z=float(zs[iz])*freq
                cz=math.cos(z)

                g = cx + cy + cz  # Schwarz P implicit
                v = 1.0 / (1.0 + math.exp(-gain * g))  # sigmoid to [0..1]
                field[ix,iy,iz]=np.float32(v)
    return field

def build_ui(parent):
    d=get_defaults()
    w=QtWidgets.QWidget(parent)
    f=QtWidgets.QFormLayout(w)
    freq=QtWidgets.QDoubleSpinBox(); freq.setRange(0.5,10.0); freq.setDecimals(2); freq.setSingleStep(0.25); freq.setValue(d["FREQ"])
    gain=QtWidgets.QDoubleSpinBox(); gain.setRange(0.2,8.0); gain.setDecimals(2); gain.setSingleStep(0.1); gain.setValue(d["GAIN"])
    f.addRow("FREQ", freq)
    f.addRow("GAIN", gain)
    def get_params():
        return dict(FREQ=float(freq.value()), GAIN=float(gain.value()))
    return w, get_params

def compute(params: dict):
    return schwarz_p_field(params["N"], params["BOUNDS"], params["FREQ"], params["GAIN"])
