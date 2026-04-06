# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id":"twisted_ribbon","name":"Twisted Ribbon (implicit)","category":"Surfaces"}


FORMULA = r"""Twisted Ribbon (implicit)
Plugin: twisted_ribbon  (twisted_ribbon.py)

Idea:
An implicit ribbon / band that twists along an axis. Useful as a clean "sculpture" field.

Concept:
- start with a band around a curve (or axis)
- apply twist as a function of position
- output a smooth field around the ribbon centerline

Notes:
- ISO controls thickness
- TWIST controls how fast the ribbon rotates
"""

def get_defaults():
    return dict(R=0.8, WIDTH=0.18, TWIST=1.0, GAIN=5.0)

@njit(parallel=True, fastmath=True)
def ribbon_field(N, bounds, R, width, twist, gain):
    N=int(N); bounds=float(bounds)
    R=float(R); width=float(width); twist=float(twist); gain=float(gain)
    field=np.zeros((N,N,N), dtype=np.float32)

    xs=np.linspace(-bounds,bounds,N).astype(np.float32)
    ys=np.linspace(-bounds,bounds,N).astype(np.float32)
    zs=np.linspace(-bounds,bounds,N).astype(np.float32)

    for ix in prange(N):
        x=float(xs[ix]) / max(bounds,1e-12)
        for iy in range(N):
            y=float(ys[iy]) / max(bounds,1e-12)
            for iz in range(N):
                z=float(zs[iz]) / max(bounds,1e-12)

                # cylindrical coordinates
                r = math.sqrt(x*x + y*y)
                ang = math.atan2(y, x)

                # target ring radius R
                dr = r - R

                # twist: rotate local frame along angle
                t = twist * 0.5 * ang
                ct = math.cos(t)
                st = math.sin(t)

                # local coordinates: (u along radial, v along z) rotated
                u = dr*ct + z*st
                v = -dr*st + z*ct

                # ribbon implicit: near v=0 and |u| <= width
                d = max(abs(v) - 0.03, abs(u) - width)

                val = 1.0 / (1.0 + math.exp(gain * d))  # high near inside
                field[ix,iy,iz]=np.float32(val)

    return field

def build_ui(parent):
    d=get_defaults()
    w=QtWidgets.QWidget(parent)
    f=QtWidgets.QFormLayout(w)

    R=QtWidgets.QDoubleSpinBox(); R.setRange(0.2,1.4); R.setDecimals(2); R.setSingleStep(0.05); R.setValue(d["R"])
    W=QtWidgets.QDoubleSpinBox(); W.setRange(0.05,0.40); W.setDecimals(3); W.setSingleStep(0.01); W.setValue(d["WIDTH"])
    T=QtWidgets.QDoubleSpinBox(); T.setRange(0.0,6.0); T.setDecimals(2); T.setSingleStep(0.25); T.setValue(d["TWIST"])
    G=QtWidgets.QDoubleSpinBox(); G.setRange(0.5,12.0); G.setDecimals(2); G.setSingleStep(0.5); G.setValue(d["GAIN"])

    f.addRow("R (ring)", R)
    f.addRow("WIDTH", W)
    f.addRow("TWIST", T)
    f.addRow("GAIN", G)

    def get_params():
        return dict(R=float(R.value()), WIDTH=float(W.value()), TWIST=float(T.value()), GAIN=float(G.value()))
    return w, get_params

def compute(params: dict):
    return ribbon_field(params["N"], params["BOUNDS"], params["R"], params["WIDTH"], params["TWIST"], params["GAIN"])
