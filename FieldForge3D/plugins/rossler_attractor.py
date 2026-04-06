# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# plugins/rossler_attractor.py
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from PyQt6 import QtWidgets

PLUGIN_META = {"id": "rossler_attractor", "name": "Rössler Attractor (density)", "category": "Chaos"}

FORMULA = r"""Rössler Attractor (density histogram)
Plugin: rossler_attractor  (rossler_attractor.py)

Differential equations
  dx/dt = -y - z
  dy/dt =  x + a*y
  dz/dt =  b + z*(x - c)

Euler integration
  x_{n+1} = x_n + dt * (-y_n - z_n)
  y_{n+1} = y_n + dt * ( x_n + a*y_n)
  z_{n+1} = z_n + dt * ( b + z_n*(x_n - c))

Density histogram
- The trajectory is normalized into the cube [-1..1]^3:
    xn = x/sx, yn = y/sy, zn = (z-10)/sz
- Voxel indices:
    ix = floor((xn+1)/2 * (N-1)), similarly iy, iz
- hist[ix,iy,iz] += 1

Smoothing (SMOOTH)
- A few passes of a 7-point stencil (center + 6 neighbors) with edge clamping.
- Makes the density look like continuous smoke instead of speckle noise.

Output mapping
  hist_norm = hist / max(hist)
  value = 1 - exp(-GAIN * hist_norm)

Tips
- Increase STEPS for denser attractor (slower).
- Increase SMOOTH for a softer look.
- If it becomes empty: reduce BOUNDS or reduce ISO.
"""


def get_defaults():
    return dict(
        A=0.2,
        B=0.2,
        C=5.7,
        STEPS=80000,
        DT=0.01,
        SMOOTH=2.5,
        GAIN=2.6,
        SEED=1,
    )

@njit(fastmath=True)
def lcg(seed: int) -> int:
    return (seed * 1664525 + 1013904223) & 0xFFFFFFFF

@njit(parallel=True, fastmath=True)
def smooth7_edge(f: np.ndarray, passes: int) -> np.ndarray:
    """
    Numba-friendly 3D smoothing bez np.pad:
    7-point stencil (center + 6 axis neighbors) s edge clamp.
    """
    N = f.shape[0]
    out = np.empty_like(f)

    for _ in range(int(passes)):
        for i in prange(N):
            im1 = i - 1
            if im1 < 0:
                im1 = 0
            ip1 = i + 1
            if ip1 >= N:
                ip1 = N - 1

            for j in range(N):
                jm1 = j - 1
                if jm1 < 0:
                    jm1 = 0
                jp1 = j + 1
                if jp1 >= N:
                    jp1 = N - 1

                for k in range(N):
                    km1 = k - 1
                    if km1 < 0:
                        km1 = 0
                    kp1 = k + 1
                    if kp1 >= N:
                        kp1 = N - 1

                    c = f[i, j, k]
                    s = (
                        c +
                        f[im1, j, k] + f[ip1, j, k] +
                        f[i, jm1, k] + f[i, jp1, k] +
                        f[i, j, km1] + f[i, j, kp1]
                    )
                    out[i, j, k] = s * (1.0 / 7.0)

        # swap buffers
        f, out = out, f

    return f

@njit(parallel=True, fastmath=True)
def rossler_density_field(N, bounds, a, b, c, steps, dt, smooth, gain, seed):
    N = int(N)
    bounds = float(bounds)
    a = float(a)
    b = float(b)
    c = float(c)
    steps = int(steps)
    dt = float(dt)
    smooth = float(smooth)
    gain = float(gain)
    seed = int(seed)

    # hard clamps (crash-proof)
    if steps < 1000:
        steps = 1000
    if steps > 80000:
        steps = 80000

    if dt < 0.001:
        dt = 0.001
    if dt > 0.05:
        dt = 0.05

    # histogram grid
    hist = np.zeros((N, N, N), dtype=np.float32)

    # init (small random)
    s = seed & 0xFFFFFFFF
    s = lcg(s); x = (s / 4294967295.0 - 0.5) * 0.2
    s = lcg(s); y = (s / 4294967295.0 - 0.5) * 0.2
    s = lcg(s); z = (s / 4294967295.0 - 0.5) * 0.2

    # burn-in
    burn = min(2000, steps // 10 + 200)
    for _ in range(burn):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dt * dx
        y += dt * dy
        z += dt * dz

    # empirical scale to fit bounds
    sx = 12.0
    sy = 12.0
    sz = 30.0

    for _ in range(steps):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dt * dx
        y += dt * dy
        z += dt * dz

        xn = x / sx
        yn = y / sy
        zn = (z - 10.0) / sz

        # numeric safety (avoid NaN/Inf explosions)
        if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z):
            break

        # keep within normalized cube [-1..1]
        if xn < -1.0 or xn > 1.0 or yn < -1.0 or yn > 1.0 or zn < -1.0 or zn > 1.0:
            continue

        ix = int((xn + 1.0) * 0.5 * (N - 1))
        iy = int((yn + 1.0) * 0.5 * (N - 1))
        iz = int((zn + 1.0) * 0.5 * (N - 1))

        hist[ix, iy, iz] += 1.0

    # smooth passes (0..6)
    passes = int(max(0.0, min(6.0, smooth)))
    f = hist
    if passes > 0:
        f = smooth7_edge(f, passes)

    # normalize -> [0..1]
        # safe parallel max: per-slice maxima + serial reduce (no data race)
    mx_slices = np.zeros(N, dtype=np.float32)

    for i in prange(N):
        m = 0.0
        for j in range(N):
            for k in range(N):
                v = float(f[i, j, k])
                if v > m:
                    m = v
        mx_slices[i] = m

    mx = 0.0
    for i in range(N):
        v = float(mx_slices[i])
        if v > mx:
            mx = v

    if mx <= 1e-12:
        return f.astype(np.float32)

    f = f / mx
    f = 1.0 - np.exp(-gain * f)
    return f.astype(np.float32)

def build_ui(parent):
    d = get_defaults()
    w = QtWidgets.QWidget(parent)
    f = QtWidgets.QFormLayout(w)

    a = QtWidgets.QDoubleSpinBox(); a.setRange(0.01, 2.0); a.setDecimals(3); a.setSingleStep(0.01); a.setValue(d["A"])
    b = QtWidgets.QDoubleSpinBox(); b.setRange(0.01, 2.0); b.setDecimals(3); b.setSingleStep(0.01); b.setValue(d["B"])
    c = QtWidgets.QDoubleSpinBox(); c.setRange(0.5, 20.0); c.setDecimals(3); c.setSingleStep(0.1); c.setValue(d["C"])
    steps = QtWidgets.QSpinBox(); steps.setRange(5000, 80000); steps.setSingleStep(2000); steps.setValue(d["STEPS"])
    dt = QtWidgets.QDoubleSpinBox(); dt.setRange(0.001, 0.05); dt.setDecimals(4); dt.setSingleStep(0.001); dt.setValue(d["DT"])
    smooth = QtWidgets.QDoubleSpinBox(); smooth.setRange(0.0, 6.0); smooth.setDecimals(2); smooth.setSingleStep(0.25); smooth.setValue(d["SMOOTH"])
    gain = QtWidgets.QDoubleSpinBox(); gain.setRange(0.2, 8.0); gain.setDecimals(2); gain.setSingleStep(0.1); gain.setValue(d["GAIN"])
    seed = QtWidgets.QSpinBox(); seed.setRange(1, 999999); seed.setValue(d["SEED"])

    f.addRow("A", a)
    f.addRow("B", b)
    f.addRow("C", c)
    f.addRow("STEPS", steps)
    f.addRow("DT", dt)
    f.addRow("SMOOTH", smooth)
    f.addRow("GAIN", gain)
    f.addRow("SEED", seed)

    def get_params():
        return dict(
            A=float(a.value()),
            B=float(b.value()),
            C=float(c.value()),
            STEPS=int(steps.value()),
            DT=float(dt.value()),
            SMOOTH=float(smooth.value()),
            GAIN=float(gain.value()),
            SEED=int(seed.value()),
        )
    return w, get_params

def compute(params: dict):
    return rossler_density_field(
        params["N"], params["BOUNDS"],
        params["A"], params["B"], params["C"],
        params["STEPS"], params["DT"],
        params["SMOOTH"], params["GAIN"], params["SEED"]
    )
