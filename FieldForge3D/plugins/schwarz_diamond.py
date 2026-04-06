# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/schwarz_diamond.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "diamond_tpms", "name": "Diamond (TPMS)", "category": "TPMS"}


FORMULA = r"""Schwarz D (Diamond TPMS)
Plugin: schwarz_diamond  (schwarz_diamond.py)

Idea:
The Schwarz D (diamond) triply-periodic minimal surface.
A symmetric TPMS with diamond-like channels.

Formula (one common approximation):
- F = sin(x)*sin(y)*sin(z) + sin(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z) + cos(x)*cos(y)*sin(z)

Mapping:
- output v = clamp(0.5 + 0.5 * F) (or a shell around F=0)

Notes:
- freq scales space
- ISO selects the shell thickness around F=0
"""

def get_defaults() -> dict:
    return dict(FREQ=3.0, MIX=1.0)

@njit(parallel=True, fastmath=True)
def diamond_field(N, bounds, freq, mix):
    N=int(N); bounds=float(bounds); freq=float(freq); mix=float(mix)
    field=np.zeros((N,N,N), dtype=np.float32)
    xs=np.linspace(-bounds,bounds,N).astype(np.float32)
    ys=np.linspace(-bounds,bounds,N).astype(np.float32)
    zs=np.linspace(-bounds,bounds,N).astype(np.float32)

    for ix in prange(N):
        x=float(xs[ix]) * freq
        sx=math.sin(x); cx=math.cos(x)
        for iy in range(N):
            y=float(ys[iy]) * freq
            sy=math.sin(y); cy=math.cos(y)
            for iz in range(N):
                z=float(zs[iz]) * freq
                sz=math.sin(z); cz=math.cos(z)

                # classic Diamond TPMS:
                # sin x sin y sin z + sin x cos y cos z + cos x sin y cos z + cos x cos y sin z = 0
                g = sx*sy*sz + sx*cy*cz + cx*sy*cz + cx*cy*sz

                # mix allows to “sharpen”/scale
                g = mix * g

                # g ~[-2..2] (rough), map
                v = 0.5 + 0.5 * (g / 2.0)
                if v < 0.0: v=0.0
                if v > 1.0: v=1.0
                field[ix,iy,iz]=np.float32(v)

    return field

def build_ui(parent):
    d=get_defaults()
    w=QtWidgets.QWidget(parent)
    f=QtWidgets.QFormLayout(w)

    spF=QtWidgets.QDoubleSpinBox(); spF.setRange(0.5, 10.0); spF.setSingleStep(0.25); spF.setDecimals(2); spF.setValue(d["FREQ"])
    spM=QtWidgets.QDoubleSpinBox(); spM.setRange(0.1, 3.0); spM.setSingleStep(0.1); spM.setDecimals(2); spM.setValue(d["MIX"])

    f.addRow("FREQ", spF)
    f.addRow("MIX", spM)

    def get_params():
        return dict(FREQ=float(spF.value()), MIX=float(spM.value()))
    return w, get_params

def compute(params: dict):
    return diamond_field(params["N"], params["BOUNDS"], params["FREQ"], params["MIX"])
