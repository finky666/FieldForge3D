# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id":"galaxy_swirl","name":"Galaxy Swirl (scalar)","category":"Art"}


FORMULA = r"""Galaxy Swirl (scalar density)
Plugin: galaxy_swirl  (galaxy_swirl.py)

Idea:
A simple procedural "galaxy" density field: a bright core + spiral arms, with a thickness
falloff along the Z axis.

Formulas:
- r = sqrt(x^2 + y^2),  θ = atan2(y, x)
- core density:    d_core = exp( -(r^2 + z^2) / core^2 )
- arms modulation: arm    = 0.5 + 0.5 * cos( arms*θ + twist*(3r) )
- z thickness:     d_z    = exp( -(z^2) / thick^2 )
- val = d_core + arm * d_z * exp(-2r)
- output: v = 1 - exp( -gain * val )

Notes:
- gain increases contrast (exponential compression)
"""

def get_defaults():
    return dict(ARMS=3.0, TWIST=4.0, CORE=0.35, THICK=0.20, GAIN=4.0)

@njit(parallel=True, fastmath=True)
def galaxy_field(N, bounds, arms, twist, core, thick, gain):
    N=int(N); bounds=float(bounds)
    arms=float(arms); twist=float(twist)
    core=float(core); thick=float(thick); gain=float(gain)

    field=np.zeros((N,N,N), dtype=np.float32)
    xs=np.linspace(-bounds,bounds,N).astype(np.float32)
    ys=np.linspace(-bounds,bounds,N).astype(np.float32)
    zs=np.linspace(-bounds,bounds,N).astype(np.float32)

    for ix in prange(N):
        x=float(xs[ix]) / max(bounds,1e-12)
        for iy in range(N):
            y=float(ys[iy]) / max(bounds,1e-12)
            rxy=math.sqrt(x*x+y*y)
            ang=math.atan2(y,x)
            for iz in range(N):
                z=float(zs[iz]) / max(bounds,1e-12)

                # core density
                dcore = math.exp(-(rxy*rxy + z*z)/(core*core))

                # spiral arms: phase depends on radius
                phase = arms*ang + twist*rxy*3.0
                arm = 0.5 + 0.5*math.cos(phase)
                # thickness falloff
                dz = math.exp(-(z*z)/(thick*thick))
                val = dcore + arm*dz*math.exp(-rxy*2.0)

                v = 1.0 - math.exp(-gain*val)
                field[ix,iy,iz]=np.float32(v)

    return field

def build_ui(parent):
    d=get_defaults()
    w=QtWidgets.QWidget(parent)
    f=QtWidgets.QFormLayout(w)

    arms=QtWidgets.QDoubleSpinBox(); arms.setRange(1.0,8.0); arms.setDecimals(2); arms.setSingleStep(0.5); arms.setValue(d["ARMS"])
    twist=QtWidgets.QDoubleSpinBox(); twist.setRange(0.0,10.0); twist.setDecimals(2); twist.setSingleStep(0.5); twist.setValue(d["TWIST"])
    core=QtWidgets.QDoubleSpinBox(); core.setRange(0.10,0.80); core.setDecimals(2); core.setSingleStep(0.05); core.setValue(d["CORE"])
    thick=QtWidgets.QDoubleSpinBox(); thick.setRange(0.05,0.60); thick.setDecimals(2); thick.setSingleStep(0.05); thick.setValue(d["THICK"])
    gain=QtWidgets.QDoubleSpinBox(); gain.setRange(0.5,12.0); gain.setDecimals(2); gain.setSingleStep(0.5); gain.setValue(d["GAIN"])

    f.addRow("ARMS", arms)
    f.addRow("TWIST", twist)
    f.addRow("CORE", core)
    f.addRow("THICK", thick)
    f.addRow("GAIN", gain)

    def get_params():
        return dict(ARMS=float(arms.value()), TWIST=float(twist.value()),
                    CORE=float(core.value()), THICK=float(thick.value()), GAIN=float(gain.value()))
    return w, get_params

def compute(params: dict):
    return galaxy_field(params["N"], params["BOUNDS"], params["ARMS"], params["TWIST"],
                        params["CORE"], params["THICK"], params["GAIN"])
