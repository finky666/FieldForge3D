# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/mandelbulb.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets


PLUGIN_META = {
    "id": "mandelbulb",
    "name": "Mandelbulb",
    "category": "Field",
}



FORMULA = r"""Mandelbulb (3D fractal, distance-ish)
Plugin: mandelbulb  (mandelbulb.py)

Concept
- 3D analogue of the Mandelbrot set using spherical coordinates.
- Iteration: z_{n+1} = z_n^POWER + c

Spherical transform (sketch)
  r = ||z||
  θ = acos(z_z / r)
  φ = atan2(z_y, z_x)

  r' = r^POWER
  θ' = θ * POWER
  φ' = φ * POWER

  z' = r' * (sinθ'cosφ', sinθ'sinφ', cosθ') + c

Field output
- Uses an escape-based estimate mapped to [0..1].
- Higher POWER changes the symmetry and "spikiness".

Tips
- Increase MAX_ITER for more detail (slower).
- If it looks empty: lower ISO or reduce BOUNDS.
"""

def get_defaults() -> dict:
    return dict(
        POWER=8.0,
        MAX_ITER=60,
        BAILOUT=2.0,
    )


@njit(parallel=True, fastmath=True)
def mandelbulb_escape_field(N, bounds, power, max_iter, bailout):
    bounds = float(bounds)
    power = float(power)
    bailout = float(bailout)
    N = int(N)
    max_iter = int(max_iter)

    field = np.zeros((N, N, N), dtype=np.float32)

    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    b2 = np.float32(bailout * bailout)

    for ix in prange(N):
        x0 = xs[ix]
        for iy in range(N):
            y0 = ys[iy]
            for iz in range(N):
                z0 = zs[iz]

                zx = np.float32(0.0)
                zy = np.float32(0.0)
                zz = np.float32(0.0)

                it = 0
                for k in range(max_iter):
                    r2 = zx * zx + zy * zy + zz * zz
                    if r2 > b2:
                        break

                    r = math.sqrt(r2)
                    if r < 1e-12:
                        theta = 0.0
                        phi = 0.0
                    else:
                        theta = math.acos(zz / r)
                        phi = math.atan2(zy, zx)

                    rp = r ** power
                    thetap = theta * power
                    phip = phi * power

                    sin_t = math.sin(thetap)
                    zx = rp * sin_t * math.cos(phip) + x0
                    zy = rp * sin_t * math.sin(phip) + y0
                    zz = rp * math.cos(thetap) + z0

                    it = k + 1

                field[ix, iy, iz] = np.float32(1.0 if it >= max_iter else (it / max_iter))

    return field


def build_ui(parent):
    d = get_defaults()

    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    sp_power = QtWidgets.QDoubleSpinBox()
    sp_power.setRange(2.0, 12.0)
    sp_power.setSingleStep(0.5)
    sp_power.setDecimals(2)
    sp_power.setValue(d["POWER"])

    sp_iter = QtWidgets.QSpinBox()
    sp_iter.setRange(10, 200)
    sp_iter.setValue(d["MAX_ITER"])

    sp_bail = QtWidgets.QDoubleSpinBox()
    sp_bail.setRange(1.5, 10.0)
    sp_bail.setSingleStep(0.5)
    sp_bail.setDecimals(2)
    sp_bail.setValue(d["BAILOUT"])

    f.addRow("POWER", sp_power)
    f.addRow("MAX_ITER", sp_iter)
    f.addRow("BAILOUT", sp_bail)

    def get_params():
        return dict(
            POWER=float(sp_power.value()),
            MAX_ITER=int(sp_iter.value()),
            BAILOUT=float(sp_bail.value()),
        )

    return w, get_params


def compute(params: dict):
    return mandelbulb_escape_field(
        params["N"],
        params["BOUNDS"],
        params["POWER"],
        params["MAX_ITER"],
        params["BAILOUT"],
    )
