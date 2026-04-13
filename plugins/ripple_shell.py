from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {
    "id": "ripple_shell",
    "name": "Ripple Shell",
    "category": "Math Art",
    "printable": True,
}

FORMULA = r"""Ripple Shell
Plugin: ripple_shell  (ripple_shell.py)

Idea:
A closed shell with radial ripples travelling over a rounded body.
Useful both as a visual showcase and as a printable mathematical ornament.

Field:
- r = sqrt(x^2 + y^2 + z^2)
- theta = atan2(y,x)
- phi = atan2(z, sqrt(x^2+y^2))
- body = r - R - a*sin(k*theta)*cos(m*phi)
- shell = exp(-gain * |body| / thickness)
"""


def get_defaults() -> dict:
    return {
        "N": 180,
        "BOUNDS": 1.5,
        "ISO": 0.56,
        "RADIUS": 0.82,
        "AMPLITUDE": 0.10,
        "WAVES": 6,
        "BANDS": 4,
        "THICK": 0.10,
        "GAIN": 3.4,
    }


@njit(parallel=True, fastmath=True)
def ripple_field(N, bounds, radius, amplitude, waves, bands, thick, gain):
    N = int(N)
    bounds = float(bounds)
    radius = float(radius)
    amplitude = float(amplitude)
    waves = int(waves)
    bands = int(bands)
    thick = float(thick)
    gain = float(gain)

    field = np.zeros((N, N, N), dtype=np.float32)
    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    for ix in prange(N):
        x = float(xs[ix])
        for iy in range(N):
            y = float(ys[iy])
            theta = math.atan2(y, x)
            xy = math.sqrt(x * x + y * y)
            for iz in range(N):
                z = float(zs[iz])
                r = math.sqrt(x * x + y * y + z * z)
                phi = math.atan2(z, max(xy, 1e-6))
                ripple = amplitude * math.sin(waves * theta) * math.cos(bands * phi)
                body = r - radius - ripple
                field[ix, iy, iz] = np.float32(math.exp(-gain * abs(body) / max(thick, 1e-6)))

    return field


def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    sp_r = QtWidgets.QDoubleSpinBox(); sp_r.setRange(0.2, 1.5); sp_r.setSingleStep(0.02); sp_r.setDecimals(2); sp_r.setValue(d["RADIUS"])
    sp_a = QtWidgets.QDoubleSpinBox(); sp_a.setRange(0.0, 0.4); sp_a.setSingleStep(0.01); sp_a.setDecimals(2); sp_a.setValue(d["AMPLITUDE"])
    sp_w = QtWidgets.QSpinBox(); sp_w.setRange(1, 16); sp_w.setValue(d["WAVES"])
    sp_b = QtWidgets.QSpinBox(); sp_b.setRange(1, 16); sp_b.setValue(d["BANDS"])
    sp_t = QtWidgets.QDoubleSpinBox(); sp_t.setRange(0.02, 0.4); sp_t.setSingleStep(0.01); sp_t.setDecimals(2); sp_t.setValue(d["THICK"])
    sp_g = QtWidgets.QDoubleSpinBox(); sp_g.setRange(0.5, 8.0); sp_g.setSingleStep(0.1); sp_g.setDecimals(2); sp_g.setValue(d["GAIN"])

    f.addRow("RADIUS", sp_r)
    f.addRow("AMPLITUDE", sp_a)
    f.addRow("WAVES", sp_w)
    f.addRow("BANDS", sp_b)
    f.addRow("THICK", sp_t)
    f.addRow("GAIN", sp_g)

    def get_params():
        return {
            "RADIUS": float(sp_r.value()),
            "AMPLITUDE": float(sp_a.value()),
            "WAVES": int(sp_w.value()),
            "BANDS": int(sp_b.value()),
            "THICK": float(sp_t.value()),
            "GAIN": float(sp_g.value()),
        }

    return w, get_params


def compute(params: dict):
    return ripple_field(
        params["N"], params["BOUNDS"], params["RADIUS"], params["AMPLITUDE"],
        params["WAVES"], params["BANDS"], params["THICK"], params["GAIN"],
    )
