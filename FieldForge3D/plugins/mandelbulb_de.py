# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/mandelbulb_de.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets


PLUGIN_META = {"id": "mandelbulb_de", "name": "Mandelbulb (DE)", "category": "Field"}



FORMULA = r"""Mandelbulb (distance estimator)
Plugin: mandelbulb_de  (mandelbulb_de.py)

Idea:
A 3D Mandelbulb rendered via a distance estimator (DE). Instead of a simple escape mask,
we estimate distance to the fractal surface and convert it to a smooth density shell.

Core:
- iterate z in spherical coordinates with power P:
  (r, θ, φ) -> (r^P, θ*P, φ*P)
- track derivative dr to compute DE

Distance estimator:
- DE ≈ 0.5 * log(r) * r / dr
- we create a shell around DE=ISO:
  v = exp( - (DE-ISO)^2 / (2*sigma^2) )   (conceptually)

Notes:
- ISO sets the shell radius; adjust with BOUNDS and ISO together
- MAX_ITER and POWER change the shape and detail
"""

def get_defaults() -> dict:
    return dict(
        POWER=8.0,
        MAX_ITER=35,
        BAILOUT=4.0,
        SCALE=1.0,
        INVERT=True,
    )


@njit(parallel=True, fastmath=True)
def mandelbulb_de_field(N, bounds, power, max_iter, bailout, scale, invert_flag):
    bounds = float(bounds)
    power = float(power)
    bailout = float(bailout)
    scale = float(scale)
    N = int(N)
    max_iter = int(max_iter)

    field = np.zeros((N, N, N), dtype=np.float32)

    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    for ix in prange(N):
        x0 = float(xs[ix])
        for iy in range(N):
            y0 = float(ys[iy])
            for iz in range(N):
                z0 = float(zs[iz])

                zx = 0.0
                zy = 0.0
                zz = 0.0

                dr = 1.0
                r = 0.0

                for _ in range(max_iter):
                    r = math.sqrt(zx*zx + zy*zy + zz*zz)
                    if r > bailout:
                        break

                    # spherical
                    if r < 1e-12:
                        theta = 0.0
                        phi = 0.0
                    else:
                        theta = math.acos(zz / r)
                        phi = math.atan2(zy, zx)

                    # derivative
                    dr = dr * (power * (r ** (power - 1.0))) + 1.0

                    # scale and rotate
                    rp = r ** power
                    thetap = theta * power
                    phip = phi * power

                    sin_t = math.sin(thetap)
                    zx = rp * sin_t * math.cos(phip) + x0
                    zy = rp * sin_t * math.sin(phip) + y0
                    zz = rp * math.cos(thetap) + z0

                # distance estimator
                # de ~ 0.5*log(r)*r/dr
                de = 0.0
                if r > 1e-12:
                    de = 0.5 * math.log(r) * r / dr

                # map de -> [0..1]
                v = de * scale
                if v < 0.0:
                    v = 0.0
                if v > 1.0:
                    v = 1.0

                if invert_flag == 1:
                    v = 1.0 - v

                field[ix, iy, iz] = np.float32(v)

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
    sp_iter.setRange(5, 120)
    sp_iter.setValue(d["MAX_ITER"])

    sp_bail = QtWidgets.QDoubleSpinBox()
    sp_bail.setRange(1.5, 12.0)
    sp_bail.setSingleStep(0.5)
    sp_bail.setDecimals(2)
    sp_bail.setValue(d["BAILOUT"])

    sp_scale = QtWidgets.QDoubleSpinBox()
    sp_scale.setRange(0.05, 10.0)
    sp_scale.setSingleStep(0.05)
    sp_scale.setDecimals(2)
    sp_scale.setValue(d["SCALE"])

    chk_inv = QtWidgets.QCheckBox("Invert")
    chk_inv.setChecked(bool(d["INVERT"]))

    f.addRow("POWER", sp_power)
    f.addRow("MAX_ITER", sp_iter)
    f.addRow("BAILOUT", sp_bail)
    f.addRow("SCALE", sp_scale)
    f.addRow("", chk_inv)

    def get_params():
        return dict(
            POWER=float(sp_power.value()),
            MAX_ITER=int(sp_iter.value()),
            BAILOUT=float(sp_bail.value()),
            SCALE=float(sp_scale.value()),
            INVERT=bool(chk_inv.isChecked()),
        )

    return w, get_params


def compute(params: dict):
    invert_flag = 1 if bool(params.get("INVERT", True)) else 0
    return mandelbulb_de_field(
        params["N"],
        params["BOUNDS"],
        params["POWER"],
        params["MAX_ITER"],
        params["BAILOUT"],
        params["SCALE"],
        invert_flag,
    )
