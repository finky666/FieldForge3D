from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QWidget, QFormLayout, QDoubleSpinBox, QSpinBox, QVBoxLayout

PLUGIN_META = {
    "id": "event_horizon_bloom",
    "name": "Event Horizon Bloom",
    "category": "Hidden / Cosmic",
    "hidden": True,
    "printable": False,
}

FORMULA = r"""Event Horizon Bloom (hidden implicit shell field)
Plugin: event_horizon_bloom  (event_horizon_bloom.py)

Idea:
A twisted toroidal shell blooms into petal-like folds around a dark central throat.
It looks halfway between a flower, a gravitational lens and a small cosmic mandala.

Formula sketch:
- Start from a torus-like implicit surface around the Z axis.
- Twist the XY plane depending on Z.
- Modulate the tube radius by a periodic petal term in polar angle.
- Convert distance-to-surface into a soft shell intensity.

Notes:
- PETALS controls rotational symmetry.
- BLOOM increases the petal amplitude.
- TWIST controls how strongly the bloom swirls through Z.
- CORE controls the size of the central dark throat.
"""


def _dspin(vmin: float, vmax: float, step: float, value: float) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(vmin, vmax)
    w.setSingleStep(step)
    w.setDecimals(3 if step < 0.1 else 2 if step < 1 else 1)
    w.setValue(value)
    return w


def _ispin(vmin: int, vmax: int, step: int, value: int) -> QSpinBox:
    w = QSpinBox()
    w.setRange(vmin, vmax)
    w.setSingleStep(step)
    w.setValue(value)
    return w


class _UI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)
        lay.addStretch(1)

        self.petals = _ispin(3, 14, 1, 7)
        self.ring_r = _dspin(0.30, 1.40, 0.01, 0.76)
        self.tube_r = _dspin(0.05, 0.70, 0.01, 0.23)
        self.bloom = _dspin(0.00, 1.20, 0.01, 0.42)
        self.twist = _dspin(-8.00, 8.00, 0.05, 2.80)
        self.core = _dspin(0.05, 0.80, 0.01, 0.24)
        self.thick = _dspin(0.01, 0.50, 0.01, 0.09)
        self.gain = _dspin(0.20, 12.0, 0.1, 3.4)

        form.addRow("PETALS", self.petals)
        form.addRow("RING_R", self.ring_r)
        form.addRow("TUBE_R", self.tube_r)
        form.addRow("BLOOM", self.bloom)
        form.addRow("TWIST", self.twist)
        form.addRow("CORE", self.core)
        form.addRow("THICK", self.thick)
        form.addRow("GAIN", self.gain)

    def values(self) -> dict:
        return {
            "PETALS": int(self.petals.value()),
            "RING_R": float(self.ring_r.value()),
            "TUBE_R": float(self.tube_r.value()),
            "BLOOM": float(self.bloom.value()),
            "TWIST": float(self.twist.value()),
            "CORE": float(self.core.value()),
            "THICK": float(self.thick.value()),
            "GAIN": float(self.gain.value()),
        }


def get_defaults() -> dict:
    return {
        "N": 220,
        "BOUNDS": 1.55,
        "ISO": 0.52,
        "PETALS": 7,
        "RING_R": 0.76,
        "TUBE_R": 0.23,
        "BLOOM": 0.42,
        "TWIST": 2.80,
        "CORE": 0.24,
        "THICK": 0.09,
        "GAIN": 3.4,
    }


def build_ui(parent=None):
    ui = _UI(parent)
    return ui, ui.values


def compute(params: dict) -> np.ndarray:
    n = int(params.get("N", 220))
    bounds = float(params.get("BOUNDS", 1.55))

    petals = int(params.get("PETALS", 7))
    ring_r = float(params.get("RING_R", 0.76))
    tube_r = max(0.03, float(params.get("TUBE_R", 0.23)))
    bloom = float(params.get("BLOOM", 0.42))
    twist = float(params.get("TWIST", 2.8))
    core = max(0.03, float(params.get("CORE", 0.24)))
    thick = max(1e-4, float(params.get("THICK", 0.09)))
    gain = max(1e-4, float(params.get("GAIN", 3.4)))

    xs = np.linspace(-bounds, bounds, n, dtype=np.float32)
    ys = np.linspace(-bounds, bounds, n, dtype=np.float32)
    zs = np.linspace(-bounds, bounds, n, dtype=np.float32)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")

    # Twist around Z so the petal pattern spirals through the volume.
    rnorm = np.sqrt(x * x + y * y + z * z) / max(bounds, 1e-6)
    ang = twist * z / max(bounds, 1e-6) + 0.8 * rnorm
    c = np.cos(ang)
    s = np.sin(ang)
    xr = c * x - s * y
    yr = s * x + c * y

    rho = np.sqrt(xr * xr + yr * yr) + 1e-6
    theta = np.arctan2(yr, xr)

    petal_term = np.cos(petals * theta + 1.8 * z / max(tube_r, 1e-6))
    petal_env = np.exp(-(z / max(1.8 * tube_r, 1e-6)) ** 2)
    tube_mod = tube_r * (1.0 + bloom * petal_term * petal_env)
    tube_mod = np.clip(tube_mod, 0.05, 0.8)

    major = ring_r * (1.0 + 0.10 * np.cos(2.0 * theta - 0.7 * z / max(bounds, 1e-6)))
    zscale = tube_r * (0.85 + 0.15 * np.cos(theta - 0.6 * z / max(bounds, 1e-6)))
    zscale = np.clip(zscale, 0.05, 0.6)

    torus = ((rho - major) / tube_mod) ** 2 + (z / zscale) ** 2 - 1.0
    shell = np.exp(-gain * np.abs(torus) / thick)

    # Dim the central region to suggest a dark horizon / throat.
    d = np.sqrt(x * x + y * y + z * z)
    throat = 1.0 - 0.55 * np.exp(-(d / core) ** 2)
    throat = np.clip(throat, 0.20, 1.0)

    # Add a faint inner halo to make the center feel more cosmic.
    halo_eq = (d / (core * 1.25)) ** 2 - 1.0
    halo = 0.32 * np.exp(-gain * np.abs(halo_eq) / max(thick * 0.85, 1e-4))

    field = np.maximum(shell * throat, halo).astype(np.float32)
    return field
