# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

from __future__ import annotations
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets
import math

PLUGIN_META = {"id":"sierpinski_cube","name":"Sierpinski Pyramid (XOR)","category":"Fractal"}


FORMULA = r"""Sierpinski Cube (recursive voids / fractal)
Plugin: sierpinski_cube  (sierpinski_cube.py)

Idea
- A 3D Sierpinski-like structure built by recursively removing sub-cubes.
- Equivalent to a Menger-style rule with cube partitions.

Field
- For each point we evaluate whether it falls into a "kept" region after LEVELS steps.
- The result is mapped into a usable [0..1] scalar field.

Tips
- More LEVELS => more detail but slower.
- Adjust ISO to select the visible shell thickness.
"""

def get_defaults():
    return dict(DEPTH=5, SOFT=1, GAIN=3.0)

@njit(fastmath=True)
def _is_sierpinski_solid(x, y, z, depth):
    # map [-1,1] -> [0,1]
    u = 0.5*(x+1.0)
    v = 0.5*(y+1.0)
    w = 0.5*(z+1.0)

    for _ in range(depth):
        u *= 2.0; v *= 2.0; w *= 2.0
        iu = int(u); iv = int(v); iw = int(w)
        # XOR-like removal: remove if parity sum is odd (classic 3D sierpinski-ish mask)
        if (iu + iv + iw) & 1:
            return 0
        u -= iu; v -= iv; w -= iw
    return 1

@njit(parallel=True, fastmath=True)
def _raw(N, bounds, depth):
    N=int(N); bounds=float(bounds); depth=int(depth)
    out = np.zeros((N,N,N), dtype=np.float32)
    for ix in prange(N):
        x = (-bounds + (2.0*bounds)*ix/(N-1)) / max(bounds,1e-12)
        for iy in range(N):
            y = (-bounds + (2.0*bounds)*iy/(N-1)) / max(bounds,1e-12)
            for iz in range(N):
                z = (-bounds + (2.0*bounds)*iz/(N-1)) / max(bounds,1e-12)
                out[ix,iy,iz] = np.float32(_is_sierpinski_solid(x,y,z,depth))
    return out

def _smooth_pass(f: np.ndarray) -> np.ndarray:
    out = f.copy()
    out[1:-1, 1:-1, 1:-1] = (
        f[1:-1, 1:-1, 1:-1] +
        f[0:-2, 1:-1, 1:-1] + f[2:  , 1:-1, 1:-1] +
        f[1:-1, 0:-2, 1:-1] + f[1:-1, 2:  , 1:-1] +
        f[1:-1, 1:-1, 0:-2] + f[1:-1, 1:-1, 2:  ]
    ) / 7.0
    return out.astype(np.float32)

def sierpinski_cube_field(N, bounds, depth, soft, gain):
    f = _raw(N, bounds, depth)
    passes = max(0, min(3, int(soft)))
    for _ in range(passes):
        f = _smooth_pass(f)
    f = 1.0 - np.exp(-float(gain)*f)
    return f.astype(np.float32)

def build_ui(parent):
    d=get_defaults()
    w=QtWidgets.QWidget(parent)
    lay=QtWidgets.QFormLayout(w)

    depth=QtWidgets.QSpinBox(); depth.setRange(1,10); depth.setValue(d["DEPTH"])
    soft=QtWidgets.QSpinBox(); soft.setRange(0,3); soft.setValue(d["SOFT"])
    gain=QtWidgets.QDoubleSpinBox(); gain.setRange(0.2,10.0); gain.setDecimals(2); gain.setSingleStep(0.1); gain.setValue(d["GAIN"])

    lay.addRow("DEPTH", depth)
    lay.addRow("SOFT", soft)
    lay.addRow("GAIN", gain)

    def get_params():
        return dict(DEPTH=int(depth.value()), SOFT=int(soft.value()), GAIN=float(gain.value()))
    return w, get_params

def compute(params: dict):
    return sierpinski_cube_field(params["N"], params["BOUNDS"], params["DEPTH"], params["SOFT"], params["GAIN"])
