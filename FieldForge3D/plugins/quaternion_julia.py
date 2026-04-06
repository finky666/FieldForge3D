# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/quaternion_julia.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "quat_julia", "name": "Quaternion Julia (3D slice)", "category": "Fractals"}


FORMULA = r"""Quaternion Julia (4D Julia slice, projected to 3D)
Plugin: quaternion_julia  (quaternion_julia.py)

Quaternion iteration
- We iterate a quaternion q in 4D:
    q_{n+1} = q_n^2 + C
- We visualize a 3D slice by fixing one component (W) and using (X,Y,Z) as space.

Field
- Escape-based field mapped to [0..1].
- MAX_ITER and bailout control the sharpness.

Tips
- Small changes in C create wildly different structures.
- If the set disappears: adjust ISO and BOUNDS first.
"""

def get_defaults() -> dict:
    return dict(
        CW=-0.20, CX=0.72, CY=0.00, CZ=0.00,
        ITERS=22,
        BAIL=4.0
    )

@njit(fastmath=True)
def qmul(a0,a1,a2,a3, b0,b1,b2,b3):
    # (w,x,y,z)
    w = a0*b0 - a1*b1 - a2*b2 - a3*b3
    x = a0*b1 + a1*b0 + a2*b3 - a3*b2
    y = a0*b2 - a1*b3 + a2*b0 + a3*b1
    z = a0*b3 + a1*b2 - a2*b1 + a3*b0
    return w,x,y,z

@njit(parallel=True, fastmath=True)
def quat_julia_field(N, bounds, cw,cx,cy,cz, iters, bail):
    N=int(N); bounds=float(bounds)
    cw=float(cw); cx=float(cx); cy=float(cy); cz=float(cz)
    iters=int(iters); bail=float(bail)

    bail2=bail*bail
    field=np.zeros((N,N,N), dtype=np.float32)

    xs=np.linspace(-bounds,bounds,N).astype(np.float32)
    ys=np.linspace(-bounds,bounds,N).astype(np.float32)
    zs=np.linspace(-bounds,bounds,N).astype(np.float32)

    for ix in prange(N):
        x0=float(xs[ix])
        for iy in range(N):
            y0=float(ys[iy])
            for iz in range(N):
                z0=float(zs[iz])

                # start quaternion q = (0, x,y,z)
                w=0.0; x=x0; y=y0; z=z0

                k=0
                for t in range(iters):
                    w,x,y,z = qmul(w,x,y,z, w,x,y,z)  # q^2
                    w += cw; x += cx; y += cy; z += cz

                    r2 = w*w + x*x + y*y + z*z
                    if r2 > bail2:
                        break
                    k=t+1

                v = 1.0 if k >= iters else (k/iters)
                field[ix,iy,iz]=np.float32(v)

    return field

def build_ui(parent):
    d=get_defaults()
    w=QtWidgets.QWidget(parent)
    f=QtWidgets.QFormLayout(w)

    spW=QtWidgets.QDoubleSpinBox(); spW.setRange(-1.5,1.5); spW.setSingleStep(0.05); spW.setDecimals(2); spW.setValue(d["CW"])
    spX=QtWidgets.QDoubleSpinBox(); spX.setRange(-1.5,1.5); spX.setSingleStep(0.05); spX.setDecimals(2); spX.setValue(d["CX"])
    spY=QtWidgets.QDoubleSpinBox(); spY.setRange(-1.5,1.5); spY.setSingleStep(0.05); spY.setDecimals(2); spY.setValue(d["CY"])
    spZ=QtWidgets.QDoubleSpinBox(); spZ.setRange(-1.5,1.5); spZ.setSingleStep(0.05); spZ.setDecimals(2); spZ.setValue(d["CZ"])

    spI=QtWidgets.QSpinBox(); spI.setRange(5,60); spI.setValue(d["ITERS"])
    spB=QtWidgets.QDoubleSpinBox(); spB.setRange(2.0,10.0); spB.setSingleStep(0.5); spB.setDecimals(2); spB.setValue(d["BAIL"])

    f.addRow("c.w", spW); f.addRow("c.x", spX); f.addRow("c.y", spY); f.addRow("c.z", spZ)
    f.addRow("ITERS", spI)
    f.addRow("BAIL", spB)

    def get_params():
        return dict(CW=float(spW.value()), CX=float(spX.value()), CY=float(spY.value()), CZ=float(spZ.value()),
                    ITERS=int(spI.value()), BAIL=float(spB.value()))
    return w, get_params

def compute(params: dict):
    return quat_julia_field(params["N"], params["BOUNDS"],
                            params["CW"], params["CX"], params["CY"], params["CZ"],
                            params["ITERS"], params["BAIL"])
