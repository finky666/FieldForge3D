# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/lorenz_density.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "lorenz", "name": "Lorenz Attractor (density)", "category": "Chaos"}


FORMULA = r"""Lorenz Attractor (density)
Plugin: lorenz_density  (lorenz_density.py)

Idea:
Simulate the Lorenz system and accumulate a 3D density histogram. This produces a
volumetric "butterfly" attractor that can be rendered as an isosurface.

ODE:
- dx/dt = σ (y - x)
- dy/dt = x (ρ - z) - y
- dz/dt = x y - β z

Density:
- positions are normalized into [-1..1]^3, then binned into a voxel grid
- hist[ix,iy,iz] += 1
- optional smoothing (few passes)
- normalize and compress: v = 1 - exp(-GAIN * hist_norm)

Notes:
- STEPS controls quality vs. time
- DT controls numerical stability (keep it small)
"""

def get_defaults() -> dict:
    return dict(
        SIGMA=10.0,
        RHO=28.0,
        BETA=2.6667,
        STEPS=60000,
        DT=0.005,
        SMOOTH=1.2,   # density smoothing
        GAIN=2.0
    )

@njit(fastmath=True)
def lorenz_step(x,y,z, sigma,rho,beta, dt):
    dx = sigma*(y-x)
    dy = x*(rho - z) - y
    dz = x*y - beta*z
    return x + dt*dx, y + dt*dy, z + dt*dz

@njit(parallel=True, fastmath=True)
def lorenz_density_field(N, bounds, sigma, rho, beta, steps, dt, smooth, gain):
    N=int(N); bounds=float(bounds)
    sigma=float(sigma); rho=float(rho); beta=float(beta)
    steps=int(steps); dt=float(dt)
    smooth=float(smooth); gain=float(gain)

    # accumulate hits in a voxel grid (coarse density)
    acc = np.zeros((N,N,N), dtype=np.float32)

    # simulate one trajectory
    x=0.1; y=0.0; z=0.0

    # scale Lorenz into our bounds
    # typical extents ~ x,y in [-20..20], z in [0..50]
    sx = bounds / 22.0
    sy = bounds / 22.0
    sz = bounds / 30.0

    for i in range(steps):
        x,y,z = lorenz_step(x,y,z, sigma,rho,beta, dt)

        vx = x*sx
        vy = y*sy
        vz = (z-25.0)*sz  # center z

        ix = int((vx + bounds) * (N-1) / (2.0*bounds))
        iy = int((vy + bounds) * (N-1) / (2.0*bounds))
        iz = int((vz + bounds) * (N-1) / (2.0*bounds))

        if 0 <= ix < N and 0 <= iy < N and 0 <= iz < N:
            acc[ix,iy,iz] += 1.0

    # normalize + smooth-ish mapping to [0..1]
    m = float(acc.max()) if acc.max() > 0 else 1.0
    field = np.zeros((N,N,N), dtype=np.float32)

    inv = 1.0 / m
    for ix in prange(N):
        for iy in range(N):
            for iz in range(N):
                v = float(acc[ix,iy,iz]) * inv
                # emphasize + soft clamp
                v = 1.0 - math.exp(-gain * (v * smooth))
                if v < 0.0: v=0.0
                if v > 1.0: v=1.0
                field[ix,iy,iz] = np.float32(v)

    return field

def build_ui(parent):
    d=get_defaults()
    w=QtWidgets.QWidget(parent)
    f=QtWidgets.QFormLayout(w)

    spS=QtWidgets.QDoubleSpinBox(); spS.setRange(0.1, 30.0); spS.setSingleStep(0.5); spS.setDecimals(2); spS.setValue(d["SIGMA"])
    spR=QtWidgets.QDoubleSpinBox(); spR.setRange(24.0, 60.0); spR.setSingleStep(0.5); spR.setDecimals(2); spR.setValue(d["RHO"])
    spB=QtWidgets.QDoubleSpinBox(); spB.setRange(0.1, 10.0); spB.setSingleStep(0.1); spB.setDecimals(4); spB.setValue(d["BETA"])

    spSteps=QtWidgets.QSpinBox(); spSteps.setRange(5000, 180000); spSteps.setSingleStep(5000); spSteps.setValue(d["STEPS"])
    spDT=QtWidgets.QDoubleSpinBox(); spDT.setRange(0.0005, 0.01); spDT.setSingleStep(0.001); spDT.setDecimals(3); spDT.setValue(d["DT"])
    spSm=QtWidgets.QDoubleSpinBox(); spSm.setRange(0.2, 5.0); spSm.setSingleStep(0.1); spSm.setDecimals(2); spSm.setValue(d["SMOOTH"])
    spG=QtWidgets.QDoubleSpinBox(); spG.setRange(0.2, 8.0); spG.setSingleStep(0.1); spG.setDecimals(2); spG.setValue(d["GAIN"])

    f.addRow("SIGMA", spS)
    f.addRow("RHO", spR)
    f.addRow("BETA", spB)
    f.addRow("STEPS", spSteps)
    f.addRow("DT", spDT)
    f.addRow("SMOOTH", spSm)
    f.addRow("GAIN", spG)

    def get_params():
        return dict(SIGMA=float(spS.value()), RHO=float(spR.value()), BETA=float(spB.value()),
                    STEPS=int(spSteps.value()), DT=float(spDT.value()),
                    SMOOTH=float(spSm.value()), GAIN=float(spG.value()))
    return w, get_params

def compute(params: dict):
    return lorenz_density_field(params["N"], params["BOUNDS"],
                                params["SIGMA"], params["RHO"], params["BETA"],
                                params["STEPS"], params["DT"],
                                params["SMOOTH"], params["GAIN"])
