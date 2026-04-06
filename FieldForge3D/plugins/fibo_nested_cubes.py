# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/fibo_nested_cubes.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "fibo_nested_cubes", "name": "Fibonacci Nested Cubes", "category": "Fibonacci"}



FORMULA = r"""Fibonacci Nested Cubes (wireframe / edges)
Plugin: fibo_nested_cubes  (fibo_nested_cubes.py)

Idea:
We place multiple nested wireframe cubes inside [-1..1]^3.
Cube scale grows by the golden ratio φ, so cube size shrinks roughly as ~ 1 / φ^lvl.
A small rotation per level creates the swirling / woven look.

Golden ratio:
- φ = (1 + sqrt(5)) / 2
- scale_0 = 1
- scale_(lvl+1) = scale_lvl * φ
- halfsize(hs) = base_hs / scale_lvl

Rotation per level (Euler-like):
(x, y, z) -> rot3(x, y, z, axk, ayk, azk)
axk = ROTX + TWIST * 0.73 * lvl
ayk = ROTY + TWIST * 0.57 * lvl
azk = ROTZ + TWIST * 1.00 * lvl

Wire thickness:
- th = THICK * hs
- th >= 1.2 * dx    (dx = 2 / (N-1)  -> minimum "voxel" thickness)
- th <= 0.30 * hs   (avoid turning the cube into a solid volume)
"""


def get_defaults() -> dict:
    return dict(
        LEVELS=12,
        THICK=0.05,
        GAIN=5.0,       # kontrast
        ROTX=0.35,
        ROTY=0.20,
        ROTZ=0.15,
        TWIST=0.25,
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
def rot3(x: float, y: float, z: float, ax: float, ay: float, az: float):
    cx = math.cos(ax)
    sx = math.sin(ax)
    y1 = cx * y - sx * z
    z1 = sx * y + cx * z

    cy = math.cos(ay)
    sy = math.sin(ay)
    x2 = cy * x + sy * z1
    z2 = -sy * x + cy * z1

    cz = math.cos(az)
    sz = math.sin(az)
    x3 = cz * x2 - sz * y1
    y3 = sz * x2 + cz * y1

    return x3, y3, z2


@njit(parallel=True, fastmath=True)
def fibo_nested_cubes_field(N, bounds, levels, thick, gain, rotx, roty, rotz, twist):
    N = int(N)
    bounds = float(bounds)
    levels = int(levels)
    thick = float(thick)
    gain = float(gain)
    rotx = float(rotx)
    roty = float(roty)
    rotz = float(rotz)
    twist = float(twist)

    field = np.zeros((N, N, N), dtype=np.float32)

    invb = 1.0 / max(bounds, 1e-12)
    xs = (np.linspace(-bounds, bounds, N) * invb).astype(np.float32)
    ys = (np.linspace(-bounds, bounds, N) * invb).astype(np.float32)
    zs = (np.linspace(-bounds, bounds, N) * invb).astype(np.float32)

    phi = 0.5 * (1.0 + math.sqrt(5.0))
    dx = 2.0 / max(N - 1, 1)
    base_hs = 0.85

    for i in prange(N):
        x0 = float(xs[i])
        for j in range(N):
            y0 = float(ys[j])
            for k in range(N):
                z0 = float(zs[k])

                v = 0.0
                scale = 1.0

                for lvl in range(levels):
                    hs = base_hs / scale
                    if hs < 2.0 * dx:
                        break

                    axk = rotx + twist * 0.73 * lvl
                    ayk = roty + twist * 0.57 * lvl
                    azk = rotz + twist * 1.00 * lvl

                    x, y, z = rot3(x0, y0, z0, axk, ayk, azk)

                    ax = abs(x)
                    ay = abs(y)
                    az = abs(z)

                    th = thick * hs
                    min_th = 1.2 * dx
                    if th < min_th:
                        th = min_th
                    max_th = 0.30 * hs
                    if th > max_th:
                        th = max_th

                    fx = 1.0 - abs(ax - hs) / th
                    fy = 1.0 - abs(ay - hs) / th
                    fz = 1.0 - abs(az - hs) / th

                    fx = smoothstep01(clamp01(fx))
                    fy = smoothstep01(clamp01(fy))
                    fz = smoothstep01(clamp01(fz))

                    ix = 1.0 - max(0.0, (ax - (hs + 0.25 * th)) / th)
                    iy = 1.0 - max(0.0, (ay - (hs + 0.25 * th)) / th)
                    iz = 1.0 - max(0.0, (az - (hs + 0.25 * th)) / th)

                    ix = smoothstep01(clamp01(ix))
                    iy = smoothstep01(clamp01(iy))
                    iz = smoothstep01(clamp01(iz))

                    e_xy = (fx if fx < fy else fy) * iz
                    e_xz = (fx if fx < fz else fz) * iy
                    e_yz = (fy if fy < fz else fz) * ix

                    edge = e_xy
                    if e_xz > edge:
                        edge = e_xz
                    if e_yz > edge:
                        edge = e_yz

                    m = ax
                    if ay > m:
                        m = ay
                    if az > m:
                        m = az
                    md = abs(m - hs)

                    if md <= (0.85 * th):
                        if edge > v:
                            v = edge

                    scale *= phi

                out = 1.0 - math.exp(-gain * v)
                if out < 0.0:
                    out = 0.0
                if out > 1.0:
                    out = 1.0

                field[i, j, k] = np.float32(out)

    return field


def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    spL = QtWidgets.QSpinBox()
    spL.setRange(2, 16)
    spL.setValue(d["LEVELS"])

    spT = QtWidgets.QDoubleSpinBox()
    spT.setRange(0.01, 0.25)
    spT.setSingleStep(0.01)
    spT.setDecimals(2)
    spT.setValue(d["THICK"])

    spG = QtWidgets.QDoubleSpinBox()
    spG.setRange(0.2, 10.0)
    spG.setSingleStep(0.1)
    spG.setDecimals(2)
    spG.setValue(d["GAIN"])

    spRX = QtWidgets.QDoubleSpinBox()
    spRX.setRange(-3.14, 3.14)
    spRX.setSingleStep(0.05)
    spRX.setDecimals(2)
    spRX.setValue(d["ROTX"])

    spRY = QtWidgets.QDoubleSpinBox()
    spRY.setRange(-3.14, 3.14)
    spRY.setSingleStep(0.05)
    spRY.setDecimals(2)
    spRY.setValue(d["ROTY"])

    spRZ = QtWidgets.QDoubleSpinBox()
    spRZ.setRange(-3.14, 3.14)
    spRZ.setSingleStep(0.05)
    spRZ.setDecimals(2)
    spRZ.setValue(d["ROTZ"])

    spTw = QtWidgets.QDoubleSpinBox()
    spTw.setRange(0.0, 1.20)
    spTw.setSingleStep(0.02)
    spTw.setDecimals(2)
    spTw.setValue(d["TWIST"])

    f.addRow("LEVELS", spL)
    f.addRow("THICK", spT)
    f.addRow("GAIN", spG)
    f.addRow("ROTX", spRX)
    f.addRow("ROTY", spRY)
    f.addRow("ROTZ", spRZ)
    f.addRow("TWIST", spTw)

    def get_params():
        return dict(
            LEVELS=int(spL.value()),
            THICK=float(spT.value()),
            GAIN=float(spG.value()),
            ROTX=float(spRX.value()),
            ROTY=float(spRY.value()),
            ROTZ=float(spRZ.value()),
            TWIST=float(spTw.value()),
        )

    return w, get_params


def compute(params: dict):
    return fibo_nested_cubes_field(
        params["N"],
        params["BOUNDS"],
        params["LEVELS"],
        params["THICK"],
        params["GAIN"],
        params["ROTX"],
        params["ROTY"],
        params["ROTZ"],
        params["TWIST"],
    )
