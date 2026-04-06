# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/gyroid_twist_tunnel.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "gyroid_twist_tunnel", "name": "Gyroid Twist Tunnel", "category": "TPMS"}



FORMULA = r"""Gyroid + Twist Tunnel (TPMS style)
Plugin: gyroid_twist_tunnel  (gyroid_twist_tunnel.py)

Gyroid (implicit surface)
  g = sin(fx)cos(fy) + sin(fy)cos(fz) + sin(fz)cos(fx)
  where fx = FREQ*x, fy = FREQ*y, fz = FREQ*z

Twist around Z
- Before evaluating g, the XY plane is rotated depending on z:
  (x,y) := rotZ(x, y, TWIST * z / r_norm)

Tunnel bias
- We add a radial term to carve a tunnel-like feature:
  r = sqrt(x^2 + y^2) / r_norm
  tun = (r - TUN_R)
  F = g + TUN_GAIN * tun

Shell thickness
- We convert the implicit F into a "band" around F=0:
  t = smoothstep( clamp(1 - |F|/THICK) )

Output mapping
  value = 1 - exp(-GAIN * t)

Tips
- Increase THICK for a wider band.
- Increase TUN_GAIN to make the tunnel dominate the shape.
"""

def get_defaults() -> dict:
    return dict(
        FREQ=2.4,        # gyroid frequency (1.0–5.0)
        TWIST=1.2,       # twist around Z (0–3)
        TUN_R=0.55,
        TUN_GAIN=1.2,
        THICK=0.10,
        GAIN=4.0,        # contrast
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
def rotz(x: float, y: float, a: float):
    ca = math.cos(a)
    sa = math.sin(a)
    return ca * x - sa * y, sa * x + ca * y


@njit(fastmath=True)
def gyroid(x: float, y: float, z: float) -> float:
    return math.sin(x) * math.cos(y) + math.sin(y) * math.cos(z) + math.sin(z) * math.cos(x)


@njit(parallel=True, fastmath=True)
def gyroid_tunnel_field(N, bounds, freq, twist, tun_r, tun_gain, thick, gain):
    N = int(N)
    bounds = float(bounds)
    freq = float(freq)
    twist = float(twist)
    tun_r = float(tun_r)
    tun_gain = float(tun_gain)
    thick = float(thick)
    gain = float(gain)

    field = np.zeros((N, N, N), dtype=np.float32)

    xs = np.linspace(-bounds, bounds, N).astype(np.float32)
    ys = np.linspace(-bounds, bounds, N).astype(np.float32)
    zs = np.linspace(-bounds, bounds, N).astype(np.float32)

    dx = (2.0 * bounds) / max(N - 1, 1)
    min_th = 1.0 * dx
    if thick < min_th:
        thick = min_th

    rnorm = max(bounds, 1e-9)

    for i in prange(N):
        x0 = float(xs[i])
        for j in range(N):
            y0 = float(ys[j])
            for k in range(N):
                z0 = float(zs[k])

                a = twist * (z0 / rnorm)
                x, y = rotz(x0, y0, a)
                z = z0

                # gyroid in world space
                gx = freq * x
                gy = freq * y
                gz = freq * z
                g = gyroid(gx, gy, gz)

                r = math.sqrt(x * x + y * y) / rnorm
                tun = (r - tun_r)

                F = g + tun_gain * tun

                ad = abs(F)
                t = 1.0 - (ad / thick)
                t = clamp01(t)
                t = smoothstep01(t)

                out = 1.0 - math.exp(-gain * t)
                field[i, j, k] = np.float32(out)

    return field


def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    spF = QtWidgets.QDoubleSpinBox(); spF.setRange(0.5, 6.0); spF.setSingleStep(0.1); spF.setDecimals(2); spF.setValue(d["FREQ"])
    spTw = QtWidgets.QDoubleSpinBox(); spTw.setRange(0.0, 5.0); spTw.setSingleStep(0.05); spTw.setDecimals(2); spTw.setValue(d["TWIST"])
    spR = QtWidgets.QDoubleSpinBox(); spR.setRange(0.05, 1.20); spR.setSingleStep(0.01); spR.setDecimals(2); spR.setValue(d["TUN_R"])
    spTG = QtWidgets.QDoubleSpinBox(); spTG.setRange(0.0, 4.0); spTG.setSingleStep(0.05); spTG.setDecimals(2); spTG.setValue(d["TUN_GAIN"])
    spT = QtWidgets.QDoubleSpinBox(); spT.setRange(0.01, 0.60); spT.setSingleStep(0.01); spT.setDecimals(2); spT.setValue(d["THICK"])
    spG = QtWidgets.QDoubleSpinBox(); spG.setRange(0.2, 10.0); spG.setSingleStep(0.1); spG.setDecimals(2); spG.setValue(d["GAIN"])

    f.addRow("FREQ", spF)
    f.addRow("TWIST", spTw)
    f.addRow("TUN_R", spR)
    f.addRow("TUN_GAIN", spTG)
    f.addRow("THICK", spT)
    f.addRow("GAIN", spG)

    def get_params():
        return dict(
            FREQ=float(spF.value()),
            TWIST=float(spTw.value()),
            TUN_R=float(spR.value()),
            TUN_GAIN=float(spTG.value()),
            THICK=float(spT.value()),
            GAIN=float(spG.value()),
        )

    return w, get_params


def compute(params: dict):
    return gyroid_tunnel_field(
        params["N"], params["BOUNDS"],
        params["FREQ"], params["TWIST"], params["TUN_R"], params["TUN_GAIN"],
        params["THICK"], params["GAIN"],
    )
