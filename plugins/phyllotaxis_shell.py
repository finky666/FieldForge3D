# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/phyllotaxis_shell.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "phyllotaxis_shell", "name": "Phyllotaxis Shell (Sunflower)", "category": "Fibonacci"}


FORMULA = r"""Phyllotaxis Shell (sunflower)
Plugin: phyllotaxis_shell  (phyllotaxis_shell.py)

Idea:
A phyllotaxis pattern wrapped into a 3D shell: points placed using the golden angle
create a natural spiral packing (sunflower / pinecone style).

Core:
- golden angle: ga = π * (3 - sqrt(5))
- for n = 0..N:
    θ = n * ga
    r = sqrt(n / N)
    point = (r*cosθ, r*sinθ, z(r))

Output:
A smooth density / field around the point set, suitable for isosurface rendering.

Notes:
- adjust ISO to control shell thickness
- higher N gives smoother results but costs more
"""

def get_defaults() -> dict:
    return dict(
        POINTS=2200,     # more points = denser seed pattern
        RADIUS=1.15,
        HEIGHT=0.65,
        TWIST=0.0,
        SEED=1,
        SIGMA=0.045,
        GAIN=3.2,        # density contrast
        JITTER=0.008
    )

@njit(fastmath=True)
def lcg(seed):
    return (seed * 1664525 + 1013904223) & 0xFFFFFFFF

@njit(parallel=True, fastmath=True)
def phyllotaxis_shell_field(N, bounds, points, radius, height, twist, seed, sigma, gain, jitter):
    N=int(N); bounds=float(bounds)
    points=int(points)
    radius=float(radius); height=float(height); twist=float(twist)
    seed=int(seed); sigma=float(sigma); gain=float(gain); jitter=float(jitter)

    field=np.zeros((N,N,N), dtype=np.float32)

    # normalized voxel coords in [-1..1]
    xs=np.linspace(-bounds,bounds,N).astype(np.float32)
    ys=np.linspace(-bounds,bounds,N).astype(np.float32)
    zs=np.linspace(-bounds,bounds,N).astype(np.float32)

    # golden angle
    phi=(1.0+math.sqrt(5.0))*0.5
    ga=2.0*math.pi*(1.0-1.0/phi)

    # centers (points) on a “shell”
    c=np.zeros((points,3), dtype=np.float32)
    s=seed & 0xFFFFFFFF

    for i in range(points):
        t = i / max(points-1, 1)  # 0..1
        a = i * ga + twist * (2.0*math.pi*t)

        # classic phyllotaxis radius ~ sqrt(t)
        r = radius * math.sqrt(t)

        # map to a “cone/shell”: z from -height..+height
        z = (t - 0.5) * 2.0 * height

        x = r * math.cos(a)
        y = r * math.sin(a)

        # tiny jitter
        if jitter > 0.0:
            s = lcg(s); jx = ((s/4294967295.0)-0.5)*2.0*jitter
            s = lcg(s); jy = ((s/4294967295.0)-0.5)*2.0*jitter
            s = lcg(s); jz = ((s/4294967295.0)-0.5)*2.0*jitter
            x += jx; y += jy; z += jz

        c[i,0]=np.float32(x)
        c[i,1]=np.float32(y)
        c[i,2]=np.float32(z)

    inv2 = 1.0 / max(2.0*sigma*sigma, 1e-12)

    # voxel loop
    for ix in prange(N):
        x = float(xs[ix]) / max(bounds,1e-9)
        for iy in range(N):
            y = float(ys[iy]) / max(bounds,1e-9)
            for iz in range(N):
                z = float(zs[iz]) / max(bounds,1e-9)

                acc = 0.0
                # sample all points (ok up to ~3k)
                for i in range(points):
                    dx = x - float(c[i,0])
                    dy = y - float(c[i,1])
                    dz = z - float(c[i,2])
                    d2 = dx*dx + dy*dy + dz*dz
                    acc += math.exp(-d2 * inv2)

                v = 1.0 - math.exp(-gain * acc)
                if v < 0.0: v=0.0
                if v > 1.0: v=1.0
                field[ix,iy,iz]=np.float32(v)

    return field

def build_ui(parent):
    d=get_defaults()
    w=QtWidgets.QWidget(parent)
    f=QtWidgets.QFormLayout(w)

    spP=QtWidgets.QSpinBox(); spP.setRange(300, 6000); spP.setSingleStep(100); spP.setValue(d["POINTS"])
    spR=QtWidgets.QDoubleSpinBox(); spR.setRange(0.3, 1.8); spR.setSingleStep(0.05); spR.setDecimals(2); spR.setValue(d["RADIUS"])
    spH=QtWidgets.QDoubleSpinBox(); spH.setRange(0.1, 1.6); spH.setSingleStep(0.05); spH.setDecimals(2); spH.setValue(d["HEIGHT"])
    spT=QtWidgets.QDoubleSpinBox(); spT.setRange(-6.0, 6.0); spT.setSingleStep(0.1); spT.setDecimals(2); spT.setValue(d["TWIST"])
    spSeed=QtWidgets.QSpinBox(); spSeed.setRange(1, 999999); spSeed.setValue(d["SEED"])
    spS=QtWidgets.QDoubleSpinBox(); spS.setRange(0.01, 0.15); spS.setSingleStep(0.005); spS.setDecimals(3); spS.setValue(d["SIGMA"])
    spG=QtWidgets.QDoubleSpinBox(); spG.setRange(0.2, 8.0); spG.setSingleStep(0.1); spG.setDecimals(2); spG.setValue(d["GAIN"])
    spJ=QtWidgets.QDoubleSpinBox(); spJ.setRange(0.0, 0.05); spJ.setSingleStep(0.002); spJ.setDecimals(3); spJ.setValue(d["JITTER"])

    f.addRow("POINTS", spP)
    f.addRow("RADIUS", spR)
    f.addRow("HEIGHT", spH)
    f.addRow("TWIST", spT)
    f.addRow("SIGMA", spS)
    f.addRow("GAIN", spG)
    f.addRow("JITTER", spJ)
    f.addRow("SEED", spSeed)

    def get_params():
        return dict(POINTS=int(spP.value()), RADIUS=float(spR.value()), HEIGHT=float(spH.value()),
                    TWIST=float(spT.value()), SIGMA=float(spS.value()),
                    GAIN=float(spG.value()), JITTER=float(spJ.value()), SEED=int(spSeed.value()))
    return w, get_params

def compute(params: dict):
    return phyllotaxis_shell_field(params["N"], params["BOUNDS"],
                                   params["POINTS"], params["RADIUS"], params["HEIGHT"],
                                   params["TWIST"], params["SEED"],
                                   params["SIGMA"], params["GAIN"], params["JITTER"])
