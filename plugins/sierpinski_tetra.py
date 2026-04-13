# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

from __future__ import annotations

import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets
import math

PLUGIN_META = {"id":"sierpinski_tetra","name":"Sierpinski Tetrahedron (fractal)","category":"Fractal"}



FORMULA = r"""Sierpinski tetra (membership + smoothing)
Plugin: sierpinski_tetra  (sierpinski_tetra.py)

Formulas:
  - works in [0,1]^3 and keeps only the tetra region x+y+z ≤ 1 1
      (x,y,z) := 2·(x,y,z)
      ix=int(x), iy=int(y), iz=int(z), potom frac part

Notes:
"""

def get_defaults():
    return dict(
        DEPTH=6,
        SOFT=1,    # 0..3
        GAIN=3.0,  # kontrast
    )


# -----------------------------
# Helpers
# -----------------------------
@njit(fastmath=True)
def _in_base_tetra(x, y, z):
    """
    Base tetrahedron inside unit cube [0,1]^3:
      x >= 0, y >= 0, z >= 0, x + y + z <= 1
    """
    return (x >= 0.0) and (y >= 0.0) and (z >= 0.0) and ((x + y + z) <= 1.0)


@njit(fastmath=True)
def _sierpinski_tetra_membership(x, y, z, depth):
    """
    Classic Sierpinski tetra via "octant fold" in barycentric-like coordinates.
    Work in [0,1]^3 and keep only the tetra region x+y+z<=1.
    At each iteration scale by 2 and reject points that land in the removed middle tetra.
    """
    # map from [-1,1] to [0,1]
    x = 0.5 * (x + 1.0)
    y = 0.5 * (y + 1.0)
    z = 0.5 * (z + 1.0)

    # outside base tetra => empty
    if not _in_base_tetra(x, y, z):
        return 0

    for _ in range(depth):
        # scale up
        x *= 2.0
        y *= 2.0
        z *= 2.0

        ix = int(x)
        iy = int(y)
        iz = int(z)

        # fractional part
        x -= ix
        y -= iy
        z -= iz

        # There are 8 sub-tetra positions in the subdivided cube.
        # We keep 4 corner tetrahedra and remove the central one(s).
        # A robust rule: remove if we land in "mixed" octants (sum of bits >= 2).
        # This creates the correct tetra recursion.
        bits = (ix & 1) + (iy & 1) + (iz & 1)
        if bits >= 2:
            return 0

        # also enforce staying inside tetra at each level
        if (x + y + z) > 1.0:
            return 0

    return 1


@njit(parallel=True, fastmath=True)
def _raw_field(N, bounds, depth):
    N = int(N)
    bounds = float(bounds)
    depth = int(depth)

    field = np.zeros((N, N, N), dtype=np.float32)

    for ix in prange(N):
        x = (-bounds + (2.0 * bounds) * (ix / (N - 1))) / max(bounds, 1e-12)
        for iy in range(N):
            y = (-bounds + (2.0 * bounds) * (iy / (N - 1))) / max(bounds, 1e-12)
            for iz in range(N):
                z = (-bounds + (2.0 * bounds) * (iz / (N - 1))) / max(bounds, 1e-12)
                field[ix, iy, iz] = np.float32(_sierpinski_tetra_membership(x, y, z, depth))

    return field


def _smooth_pass(f: np.ndarray) -> np.ndarray:
    # 7-point stencil (bez pad)
    out = f.copy()
    out[1:-1, 1:-1, 1:-1] = (
        f[1:-1, 1:-1, 1:-1] +
        f[0:-2, 1:-1, 1:-1] + f[2:  , 1:-1, 1:-1] +
        f[1:-1, 0:-2, 1:-1] + f[1:-1, 2:  , 1:-1] +
        f[1:-1, 1:-1, 0:-2] + f[1:-1, 1:-1, 2:  ]
    ) / 7.0
    return out.astype(np.float32)


def sierpinski_tetra_field(N, bounds, depth, soft, gain):
    field = _raw_field(N, bounds, depth)

    passes = max(0, min(3, int(soft)))
    for _ in range(passes):
        field = _smooth_pass(field)

    field = 1.0 - np.exp(-float(gain) * field)
    return field.astype(np.float32)


def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    depth = QtWidgets.QSpinBox()
    depth.setRange(1, 10)
    depth.setValue(d["DEPTH"])

    soft = QtWidgets.QSpinBox()
    soft.setRange(0, 3)
    soft.setValue(d["SOFT"])

    gain = QtWidgets.QDoubleSpinBox()
    gain.setRange(0.2, 10.0)
    gain.setDecimals(2)
    gain.setSingleStep(0.1)
    gain.setValue(d["GAIN"])

    f.addRow("DEPTH", depth)
    f.addRow("SOFT", soft)
    f.addRow("GAIN", gain)

    def get_params():
        return dict(
            DEPTH=int(depth.value()),
            SOFT=int(soft.value()),
            GAIN=float(gain.value()),
        )

    return w, get_params


def compute(params: dict):
    return sierpinski_tetra_field(
        params["N"], params["BOUNDS"],
        params["DEPTH"], params["SOFT"], params["GAIN"]
    )
