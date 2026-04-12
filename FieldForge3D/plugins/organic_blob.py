from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {
    "id": "organic_blob",
    "name": "Organic Blob",
    "category": "Math Art",
    "printable": True,
}

FORMULA = r"""Organic Blob (metaball-like field)
Plugin: organic_blob  (organic_blob.py)

Idea:
A smooth closed blob made from a few interacting radial fields.
The result sits somewhere between metaballs and a sculpted cell / drop.

Field:
- Each seed contributes w_i = exp(-k * ||p-c_i||^2)
- The seeds drift using simple sine/cosine offsets
- Final field is clamped into [0,1]
"""


def get_defaults() -> dict:
    return {
        "N": 180,
        "BOUNDS": 1.4,
        "ISO": 0.52,
        "STRETCH": 1.00,
        "WOBBLE": 0.22,
        "SHARPNESS": 3.40,
    }


@njit(parallel=True, fastmath=True)
def blob_field(N, bounds, stretch, wobble, sharpness):
    N = int(N)
    bounds = float(bounds)
    stretch = float(stretch)
    wobble = float(wobble)
    sharpness = float(sharpness)

    field = np.zeros((N, N, N), dtype=np.float32)
    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    centers = np.array([
        (-0.42,  0.00,  0.00),
        ( 0.42,  0.00,  0.00),
        ( 0.00,  0.32,  0.00),
        ( 0.00, -0.26,  0.14),
        ( 0.00,  0.00, -0.30),
    ], dtype=np.float32)

    for ix in prange(N):
        x = float(xs[ix])
        for iy in range(N):
            y = float(ys[iy])
            for iz in range(N):
                z = float(zs[iz])
                total = 0.0
                for i in range(centers.shape[0]):
                    cx = centers[i, 0] + wobble * math.sin((i + 1.0) * 1.4)
                    cy = centers[i, 1] + wobble * 0.8 * math.cos((i + 2.0) * 1.1)
                    cz = centers[i, 2] + wobble * 0.6 * math.sin((i + 3.0) * 1.9)
                    dx = x - cx
                    dy = (y - cy) / max(stretch, 1e-6)
                    dz = z - cz
                    total += math.exp(-sharpness * (dx * dx + dy * dy + dz * dz))
                v = min(max(total / 2.0, 0.0), 1.0)
                field[ix, iy, iz] = np.float32(v)
    return field


def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    sp_stretch = QtWidgets.QDoubleSpinBox(); sp_stretch.setRange(0.5, 1.8); sp_stretch.setSingleStep(0.05); sp_stretch.setDecimals(2); sp_stretch.setValue(d["STRETCH"])
    sp_wobble = QtWidgets.QDoubleSpinBox(); sp_wobble.setRange(0.0, 0.5); sp_wobble.setSingleStep(0.01); sp_wobble.setDecimals(2); sp_wobble.setValue(d["WOBBLE"])
    sp_sharp = QtWidgets.QDoubleSpinBox(); sp_sharp.setRange(1.0, 8.0); sp_sharp.setSingleStep(0.1); sp_sharp.setDecimals(2); sp_sharp.setValue(d["SHARPNESS"])

    f.addRow("STRETCH", sp_stretch)
    f.addRow("WOBBLE", sp_wobble)
    f.addRow("SHARPNESS", sp_sharp)

    def get_params():
        return {
            "STRETCH": float(sp_stretch.value()),
            "WOBBLE": float(sp_wobble.value()),
            "SHARPNESS": float(sp_sharp.value()),
        }

    return w, get_params


def compute(params: dict):
    return blob_field(params["N"], params["BOUNDS"], params["STRETCH"], params["WOBBLE"], params["SHARPNESS"])
