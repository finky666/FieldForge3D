from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PyQt6.QtWidgets import QWidget, QFormLayout, QDoubleSpinBox, QVBoxLayout

PLUGIN_META = {
    "id": "dancing_eggs",
    "name": "Dancing Eggs",
    "category": "Fun / Showcase",
    "hidden": False,
    "printable": True,
}

FORMULA = r"""Dancing Eggs (implicit shell field)
Plugin: dancing_eggs  (dancing_eggs.py)

Idea:
Two soft egg-like bodies lean toward each other as if frozen in motion.
The scene is static, but the relative tilt, twist, and phase make them look
like they are orbiting in a slow dance.

Formula:
- Start from an ellipsoid field:
    E(x,y,z) = (x/a)^2 + (y/b)^2 + (z/c)^2 - 1
- Add an egg-like vertical asymmetry:
    x := x / (1 + egg * z)
    y := y / (1 + egg * z)
- Build two copies with different centers and opposite rotations.
- Convert distance-to-surface into a smooth shell intensity:
    s_i = exp( -gain * |E_i| / thick )
- Final field is the union of both shells:
    F = max(s_1, s_2)
"""


@dataclass
class _Params:
    SEPARATION: float = 1.20
    TILT: float = 12.0
    TWIST: float = 8.0
    PHASE: float = 0.0
    EGGNESS: float = 0.20
    SLIMNESS: float = 0.90
    THICK: float = 0.085
    GAIN: float = 3.0
    BOB: float = 0.06


class _UI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)
        lay.addStretch(1)

        self.sep = _dspin(0.10, 3.00, 0.01, 1.20)
        self.tilt = _dspin(-75.0, 75.0, 1.0, 12.0)
        self.twist = _dspin(-180.0, 180.0, 1.0, 8.0)
        self.phase = _dspin(-180.0, 180.0, 1.0, 0.0)
        self.egg = _dspin(0.00, 0.80, 0.01, 0.20)
        self.slim = _dspin(0.30, 1.50, 0.01, 0.90)
        self.thick = _dspin(0.01, 0.50, 0.01, 0.085)
        self.gain = _dspin(0.20, 12.0, 0.1, 3.0)
        self.bob = _dspin(-1.00, 1.00, 0.01, 0.06)

        form.addRow("SEPARATION", self.sep)
        form.addRow("TILT", self.tilt)
        form.addRow("TWIST", self.twist)
        form.addRow("PHASE", self.phase)
        form.addRow("EGGNESS", self.egg)
        form.addRow("SLIMNESS", self.slim)
        form.addRow("THICK", self.thick)
        form.addRow("GAIN", self.gain)
        form.addRow("BOB", self.bob)

    def values(self) -> dict:
        return {
            "SEPARATION": float(self.sep.value()),
            "TILT": float(self.tilt.value()),
            "TWIST": float(self.twist.value()),
            "PHASE": float(self.phase.value()),
            "EGGNESS": float(self.egg.value()),
            "SLIMNESS": float(self.slim.value()),
            "THICK": float(self.thick.value()),
            "GAIN": float(self.gain.value()),
            "BOB": float(self.bob.value()),
        }


def _dspin(vmin: float, vmax: float, step: float, value: float) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(vmin, vmax)
    w.setSingleStep(step)
    w.setDecimals(3 if step < 0.1 else 2 if step < 1 else 1)
    w.setValue(value)
    return w


def get_defaults() -> dict:
    return {
        "N": 180,
        "BOUNDS": 1.6,
        "ISO": 0.58,
        "SEPARATION": 1.20,
        "TILT": 12.0,
        "TWIST": 8.0,
        "PHASE": 0.0,
        "EGGNESS": 0.20,
        "SLIMNESS": 0.90,
        "THICK": 0.085,
        "GAIN": 3.0,
        "BOB": 0.06,
    }


def build_ui(parent=None):
    ui = _UI(parent)
    return ui, ui.values


def compute(params: dict) -> np.ndarray:
    n = int(params.get("N", 180))
    bounds = float(params.get("BOUNDS", 1.6))

    separation = float(params.get("SEPARATION", 1.20))
    tilt_deg = float(params.get("TILT", 12.0))
    twist_deg = float(params.get("TWIST", 8.0))
    phase_deg = float(params.get("PHASE", 0.0))
    eggness = float(params.get("EGGNESS", 0.20))
    slimness = float(params.get("SLIMNESS", 0.90))
    thick = max(1e-4, float(params.get("THICK", 0.085)))
    gain = max(1e-4, float(params.get("GAIN", 3.0)))
    bob = float(params.get("BOB", 0.06))

    xs = np.linspace(-bounds, bounds, n, dtype=np.float32)
    ys = np.linspace(-bounds, bounds, n, dtype=np.float32)
    zs = np.linspace(-bounds, bounds, n, dtype=np.float32)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")

    phase = np.deg2rad(phase_deg)
    tilt = np.deg2rad(tilt_deg)
    twist = np.deg2rad(twist_deg)

    cx1 = -0.5 * separation * np.cos(phase)
    cy1 = -0.16 * separation * np.sin(phase)
    cz1 = bob * np.sin(phase)

    cx2 = +0.5 * separation * np.cos(phase)
    cy2 = +0.16 * separation * np.sin(phase)
    cz2 = -bob * np.sin(phase)

    s1 = _egg_shell(
        x, y, z,
        cx=cx1, cy=cy1, cz=cz1,
        rx=0.58 * slimness, ry=0.58 * slimness, rz=0.72,
        tilt_y=+tilt, twist_z=+twist,
        eggness=eggness, thick=thick, gain=gain,
    )
    s2 = _egg_shell(
        x, y, z,
        cx=cx2, cy=cy2, cz=cz2,
        rx=0.58 * slimness, ry=0.58 * slimness, rz=0.72,
        tilt_y=-tilt, twist_z=-twist,
        eggness=eggness, thick=thick, gain=gain,
    )

    return np.maximum(s1, s2).astype(np.float32)


def _egg_shell(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, *,
    cx: float, cy: float, cz: float,
    rx: float, ry: float, rz: float,
    tilt_y: float, twist_z: float,
    eggness: float, thick: float, gain: float,
) -> np.ndarray:
    x0 = x - cx
    y0 = y - cy
    z0 = z - cz

    x1, y1, z1 = _rot_z(x0, y0, z0, twist_z)
    x2, y2, z2 = _rot_y(x1, y1, z1, tilt_y)

    zn = z2 / max(rz, 1e-6)
    radial = 1.0 - 0.35 * eggness * zn
    radial += 0.040 * np.exp(-((zn + 0.22) / 0.58) ** 2)
    radial -= 0.018 * np.exp(-((zn - 0.72) / 0.34) ** 2)
    radial = np.clip(radial, 0.82, 1.16)

    xe = x2 / radial
    ye = y2 / radial
    ze = z2

    e = (xe / rx) ** 2 + (ye / ry) ** 2 + (ze / rz) ** 2 - 1.0
    return np.exp(-gain * np.abs(e) / thick)


def _rot_z(x: np.ndarray, y: np.ndarray, z: np.ndarray, a: float):
    c = np.cos(a)
    s = np.sin(a)
    xr = c * x - s * y
    yr = s * x + c * y
    return xr, yr, z


def _rot_y(x: np.ndarray, y: np.ndarray, z: np.ndarray, a: float):
    c = np.cos(a)
    s = np.sin(a)
    xr = c * x + s * z
    zr = -s * x + c * z
    return xr, y, zr
