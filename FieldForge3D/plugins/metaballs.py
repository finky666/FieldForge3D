# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/metaballs.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets


PLUGIN_META = {"id": "metaballs", "name": "Metaballs", "category": "Field"}



FORMULA = r"""Metaballs (Σ 1/r²)
Plugin: metaballs  (metaballs.py)

Idea:
A smooth implicit surface created by summing radial fields from several centers.

Formula:
- s(p) = Σ_i w_i / (||p - c_i||^2 + ε)
- output: v = 1 - exp(-STRENGTH * s)

Notes:
- BALLS controls number of centers
- STRENGTH increases contrast (stronger "blobs")
"""

def get_defaults() -> dict:
    return dict(
        STRENGTH=1.0,
        BALLS=4,          # 2..8
        SEED=1,
    )


@njit(parallel=True, fastmath=True)
def metaballs_field(N, bounds, strength, balls, seed):
    bounds = float(bounds)
    strength = float(strength)
    N = int(N)
    balls = int(balls)
    seed = int(seed)

    field = np.zeros((N, N, N), dtype=np.float32)

    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    c = np.zeros((balls, 3), dtype=np.float32)
    w = np.zeros((balls,), dtype=np.float32)

    for i in range(balls):
        a = (i + 1) * 2.0 * math.pi / balls
        r = 0.55 + 0.08 * (seed % 5)
        c[i, 0] = np.float32(r * math.cos(a))
        c[i, 1] = np.float32(r * math.sin(a))
        c[i, 2] = np.float32(0.15 * math.sin(a * (1 + (seed % 3))))
        w[i] = np.float32(1.0 - 0.07 * (i % 3))

    eps = np.float32(1e-6)

    for ix in prange(N):
        x = xs[ix]
        for iy in range(N):
            y = ys[iy]
            for iz in range(N):
                z = zs[iz]
                s = 0.0
                for i in range(balls):
                    dx = x - c[i, 0]
                    dy = y - c[i, 1]
                    dz = z - c[i, 2]
                    r2 = dx*dx + dy*dy + dz*dz + eps
                    s += w[i] / r2

                # map: 1 - exp(-k*s) -> [0..1]
                v = 1.0 - math.exp(-strength * s)
                if v < 0.0:
                    v = 0.0
                elif v > 1.0:
                    v = 1.0
                field[ix, iy, iz] = np.float32(v)

    return field


def build_ui(parent):
    d = get_defaults()

    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    sp_strength = QtWidgets.QDoubleSpinBox()
    sp_strength.setRange(0.05, 10.0)
    sp_strength.setSingleStep(0.1)
    sp_strength.setDecimals(2)
    sp_strength.setValue(d["STRENGTH"])

    sp_balls = QtWidgets.QSpinBox()
    sp_balls.setRange(2, 8)
    sp_balls.setValue(d["BALLS"])

    sp_seed = QtWidgets.QSpinBox()
    sp_seed.setRange(0, 999)
    sp_seed.setValue(d["SEED"])

    f.addRow("STRENGTH", sp_strength)
    f.addRow("BALLS", sp_balls)
    f.addRow("SEED", sp_seed)

    def get_params():
        return dict(
            STRENGTH=float(sp_strength.value()),
            BALLS=int(sp_balls.value()),
            SEED=int(sp_seed.value()),
        )

    return w, get_params


def compute(params: dict):
    balls = int(params.get("BALLS", 4))
    if balls < 2:
        balls = 2
    elif balls > 64:
        balls = 64  # hard cap for safety (fuzz/external calls)

    strength = float(params.get("STRENGTH", 1.0))
    if strength < 0.0:
        strength = 0.0

    seed = int(params.get("SEED", 1))

    return metaballs_field(
        params["N"],
        params["BOUNDS"],
        strength,
        balls,
        seed,
    )
