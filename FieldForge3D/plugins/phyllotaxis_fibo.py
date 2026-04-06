# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/phyllotaxis_fibo.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "phyllotaxis", "name": "Phyllotaxis (Fibonacci)", "category": "Fibonacci"}


FORMULA = r"""Phyllotaxis (Fibonacci spiral field)
Plugin: phyllotaxis_fibo  (phyllotaxis_fibo.py)

Idea
- A sunflower-like distribution using the golden angle.
- Points are placed on a spiral with approximately uniform packing.

Golden angle
  γ = 2π * (1 - 1/φ)  ≈ 137.507°
  θ_k = k * γ
  r_k = sqrt(k / K) * R

Field
- Each point contributes a smooth radial kernel.
- Combined points form shells / seeds / spiral patterns.

Tips
- Increase COUNT for denser packing.
- Increase GAIN for higher contrast.
"""

def get_defaults() -> dict:
    return dict(
        POINTS=900,
        TURNS=18.0,
        RADIUS=1.0,
        SIGMA=0.06,
        STRENGTH=2.2,
        SEED=1,
    )

@njit(fastmath=True)
def lcg(seed):
    # tiny rng (deterministic)
    return (seed * 1664525 + 1013904223) & 0xFFFFFFFF

@njit(parallel=True, fastmath=True)
def phyllo_field(N, bounds, points, turns, radius, sigma, strength, seed):
    N=int(N); bounds=float(bounds)
    points=int(points); turns=float(turns); radius=float(radius)
    sigma=float(sigma); strength=float(strength)
    seed=int(seed)

    field=np.zeros((N,N,N), dtype=np.float32)
    xs=np.linspace(-bounds,bounds,N).astype(np.float32)
    ys=np.linspace(-bounds,bounds,N).astype(np.float32)
    zs=np.linspace(-bounds,bounds,N).astype(np.float32)

    # golden angle
    phi = (1.0 + math.sqrt(5.0))*0.5
    ga = 2.0*math.pi*(1.0 - 1.0/phi)

    # precompute centers
    c = np.zeros((points,3), dtype=np.float32)
    s = seed & 0xFFFFFFFF
    for i in range(points):
        # phyllotaxis in xy + helix in z
        a = i * ga
        r = radius * math.sqrt(i / max(points-1, 1))
        x = r * math.cos(a)
        y = r * math.sin(a)
        z = ( (i / max(points-1,1)) - 0.5 ) * 2.0
        z *= (turns / max(1.0, turns))  # scale-ish

        # tiny jitter
        s = lcg(s); jx = ((s/4294967295.0)-0.5)*0.02
        s = lcg(s); jy = ((s/4294967295.0)-0.5)*0.02
        s = lcg(s); jz = ((s/4294967295.0)-0.5)*0.02

        c[i,0]=np.float32(x + jx)
        c[i,1]=np.float32(y + jy)
        c[i,2]=np.float32(z + jz)

    inv2 = 1.0 / max(2.0*sigma*sigma, 1e-12)

    # voxel loop
    for ix in prange(N):
        x=float(xs[ix]) / max(bounds,1e-6)
        for iy in range(N):
            y=float(ys[iy]) / max(bounds,1e-6)
            for iz in range(N):
                z=float(zs[iz]) / max(bounds,1e-6)

                acc = 0.0
                # sample subset for speed (still looks great)
                step = 1 if points <= 1200 else 2
                for i in range(0, points, step):
                    dx = x - float(c[i,0])
                    dy = y - float(c[i,1])
                    dz = z - float(c[i,2])
                    d2 = dx*dx + dy*dy + dz*dz
                    acc += math.exp(-d2 * inv2)

                v = 1.0 - math.exp(-strength * acc)
                if v < 0.0: v=0.0
                if v > 1.0: v=1.0
                field[ix,iy,iz]=np.float32(v)

    return field

def build_ui(parent):
    d=get_defaults()
    w=QtWidgets.QWidget(parent)
    f=QtWidgets.QFormLayout(w)

    spP=QtWidgets.QSpinBox(); spP.setRange(200, 6000); spP.setSingleStep(100); spP.setValue(d["POINTS"])
    spT=QtWidgets.QDoubleSpinBox(); spT.setRange(1.0, 60.0); spT.setSingleStep(1.0); spT.setDecimals(1); spT.setValue(d["TURNS"])
    spR=QtWidgets.QDoubleSpinBox(); spR.setRange(0.2, 1.8); spR.setSingleStep(0.05); spR.setDecimals(2); spR.setValue(d["RADIUS"])
    spS=QtWidgets.QDoubleSpinBox(); spS.setRange(0.02, 0.20); spS.setSingleStep(0.01); spS.setDecimals(2); spS.setValue(d["SIGMA"])
    spK=QtWidgets.QDoubleSpinBox(); spK.setRange(0.2, 6.0); spK.setSingleStep(0.1); spK.setDecimals(2); spK.setValue(d["STRENGTH"])
    spSeed=QtWidgets.QSpinBox(); spSeed.setRange(1, 999999); spSeed.setValue(d["SEED"])

    f.addRow("POINTS", spP)
    f.addRow("TURNS", spT)
    f.addRow("RADIUS", spR)
    f.addRow("SIGMA", spS)
    f.addRow("STRENGTH", spK)
    f.addRow("SEED", spSeed)

    def get_params():
        return dict(POINTS=int(spP.value()), TURNS=float(spT.value()), RADIUS=float(spR.value()),
                    SIGMA=float(spS.value()), STRENGTH=float(spK.value()), SEED=int(spSeed.value()))
    return w, get_params

def compute(params: dict):
    return phyllo_field(params["N"], params["BOUNDS"],
                        params["POINTS"], params["TURNS"], params["RADIUS"],
                        params["SIGMA"], params["STRENGTH"], params["SEED"])
