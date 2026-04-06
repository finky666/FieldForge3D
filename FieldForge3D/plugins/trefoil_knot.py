# FieldForge 3D (FieldForge3D)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file

# plugins/trefoil_knot.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "trefoil_knot", "name": "Trefoil Knot (implicit tube)", "category": "Field"}

FORMULA = r"""Trefoil knot (implicit tube)
Plugin: trefoil_knot

We build an implicit "tube around a curve".
Curve (trefoil) parameterization:
  x(t) = (2 + cos(3t)) * cos(2t)
  y(t) = (2 + cos(3t)) * sin(2t)
  z(t) = sin(3t)
Then for each point P we approximate distance to the curve by sampling t.
Field = 1 - smoothstep(dist / thickness).

Notes:
- This is a sampled distance field (not exact). Increase SAMPLES for smoother tube.
- Keep N moderate; SAMPLES makes it heavier.
"""

def get_defaults() -> dict:
    return dict(
        SCALE=0.55,      # overall knot size inside bounds
        THICK=0.12,      # tube radius (in world units)
        SAMPLES=96,      # samples along curve for distance approx
        SOFT=0.20,       # softness of tube boundary
    )

@njit(fastmath=True)
def _trefoil_pos(t: float):
    # trefoil-like curve
    x = (2.0 + math.cos(3.0*t)) * math.cos(2.0*t)
    y = (2.0 + math.cos(3.0*t)) * math.sin(2.0*t)
    z = math.sin(3.0*t)
    return x, y, z

@njit(parallel=True, fastmath=True)
def trefoil_field(N, bounds, scale, thick, samples, soft):
    N = int(N)
    bounds = float(bounds)
    scale = float(scale)
    thick = float(thick)
    soft = float(soft)
    samples = int(samples)

    field = np.zeros((N, N, N), dtype=np.float32)

    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    # precompute curve samples
    pts = np.zeros((samples, 3), dtype=np.float32)
    for i in range(samples):
        t = (2.0 * math.pi) * (i / samples)
        x, y, z = _trefoil_pos(t)
        pts[i, 0] = np.float32(x * scale)
        pts[i, 1] = np.float32(y * scale)
        pts[i, 2] = np.float32(z * scale)

    inv_soft = 1.0 / max(1e-6, soft)

    for ix in prange(N):
        x = float(xs[ix])
        for iy in range(N):
            y = float(ys[iy])
            for iz in range(N):
                z = float(zs[iz])

                # distance to sampled curve points
                d2 = 1e30
                for k in range(samples):
                    dx = x - float(pts[k,0])
                    dy = y - float(pts[k,1])
                    dz = z - float(pts[k,2])
                    dd = dx*dx + dy*dy + dz*dz
                    if dd < d2:
                        d2 = dd
                d = math.sqrt(d2)

                # tube profile: inside thick => high
                # soft edge: smooth falloff
                a = (d - thick) * inv_soft  # negative inside
                # map to [0..1] using smoothstep-ish
                if a <= -1.0:
                    v = 1.0
                elif a >= 1.0:
                    v = 0.0
                else:
                    # smoothstep on (1-a)/2
                    u = 0.5 * (1.0 - a)
                    v = u*u*(3.0 - 2.0*u)

                field[ix, iy, iz] = np.float32(v)

    return field

def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    sp_scale = QtWidgets.QDoubleSpinBox()
    sp_scale.setRange(0.10, 1.50)
    sp_scale.setDecimals(3)
    sp_scale.setSingleStep(0.05)
    sp_scale.setValue(d["SCALE"])

    sp_thick = QtWidgets.QDoubleSpinBox()
    sp_thick.setRange(0.01, 0.60)
    sp_thick.setDecimals(3)
    sp_thick.setSingleStep(0.01)
    sp_thick.setValue(d["THICK"])

    sp_soft = QtWidgets.QDoubleSpinBox()
    sp_soft.setRange(0.01, 0.60)
    sp_soft.setDecimals(3)
    sp_soft.setSingleStep(0.01)
    sp_soft.setValue(d["SOFT"])

    sp_samples = QtWidgets.QSpinBox()
    sp_samples.setRange(24, 360)
    sp_samples.setSingleStep(12)
    sp_samples.setValue(int(d["SAMPLES"]))

    f.addRow("SCALE", sp_scale)
    f.addRow("THICK", sp_thick)
    f.addRow("SOFT", sp_soft)
    f.addRow("SAMPLES", sp_samples)

    def get_params():
        return dict(
            SCALE=float(sp_scale.value()),
            THICK=float(sp_thick.value()),
            SOFT=float(sp_soft.value()),
            SAMPLES=int(sp_samples.value()),
        )

    return w, get_params

def compute(params: dict):
    return trefoil_field(
        params["N"],
        params["BOUNDS"],
        params.get("SCALE", 0.55),
        params.get("THICK", 0.12),
        params.get("SAMPLES", 96),
        params.get("SOFT", 0.20),
    )
