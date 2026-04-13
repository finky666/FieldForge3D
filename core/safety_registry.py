# FieldForge 3D (3D Field Workbench)
# Copyright (c) 2026 Tibor Cefan
# MIT License - see LICENSE file
# Credits: Developed and maintained by Tibor Cefan (finky666). Assisted by ChatGPT (Majka / SuPyWomen).

# core/safety_registry.py
# Central safety caps per plugin_id (no plugin edits required).
# Used to clamp N and to set per-plugin mesh polygon targets.

GLOBAL_LIMITS = {
    'MAX_N': 520,
    'MAX_POLYS': 4_000_000,
    'HARD_STOP': 18_000_000,
}

# Per-plugin overrides. Any missing key falls back to GLOBAL_LIMITS.
PLUGIN_LIMITS = {
    'trefoil_knot': {'MAX_N': 240, 'MAX_POLYS': 3000000},
    'heart_implicit': {'MAX_N': 360, 'MAX_POLYS': 3500000},
    'lorenz': {'MAX_N': 320, 'MAX_POLYS': 3000000},
    'mandelbox_like': {'MAX_N': 210, 'MAX_POLYS': 2200000, 'HARD_STOP': 10000000},
    'mandelbulb': {'MAX_N': 240, 'MAX_POLYS': 2800000},
    'mandelbulb_de': {'MAX_N': 220, 'MAX_POLYS': 2800000},
    'menger_sponge': {'MAX_N': 220, 'MAX_POLYS': 2000000},
    'rossler_attractor': {'MAX_N': 320, 'MAX_POLYS': 3000000},
}

# ------------------------------
# Parameter clamps (CPU / stability)
# ------------------------------
# These clamps are enforced at runtime inside app.py right after reading UI params.
# Goal: avoid settings that are known to freeze or hard-crash (Numba/VTK) on typical PCs.
#
# NOTE:
# - UI widgets may still allow wider ranges (plugin-local). Runtime clamp is the final guard.
# - If you want stricter UI ranges, also tighten them inside each plugin's build_ui().
#
# Format: { "PARAM": (min, max) }
GLOBAL_PARAM_CLAMPS = {
    # Chaos/density sims
    "STEPS": (1_000, 200_000),
    "DT": (0.0005, 0.01),

    # Fractals
    "MAX_ITER": (1, 120),
    "ITERS": (1, 120),
    "DEPTH": (1, 10),

    # Generic recursion / counts
    "LEVELS": (1, 20),
}

# Per-plugin overrides: { plugin_id: { "PARAM": (min, max) } }
PLUGIN_PARAM_CLAMPS = {
    # Lorenz: below ~24.74 the classic chaos disappears and the density tends to collapse
    # into a fat blob which can explode polygon count and freeze VTK.
    "lorenz": {
        "RHO": (24.0, 60.0),
        "STEPS": (5_000, 180_000),
        "DT": (0.0005, 0.01),
        "GAIN": (0.2, 4.0),
        "SMOOTH": (0.2, 3.0),
    },
    # Quaternion Julia / Mandelbox-like can become extremely heavy with high iters
    "quaternion_julia": {"ITERS": (1, 100)},
    "mandelbox_like": {"ITERS": (1, 120)},
    "menger_sponge": {"DEPTH": (1, 9)},
    "mandelbulb": {"MAX_ITER": (1, 100)},
    "mandelbulb_de": {"MAX_ITER": (1, 100)},
}

def clamp_params(plugin_id: str, params: dict) -> tuple[dict, list[tuple[str, float, float]]]:
    """Clamp known-dangerous params to safe bounds.

    Returns: (new_params, changes) where changes is list of (key, old, new).
    """
    pid = str(plugin_id or "")
    out = dict(params or {})
    changes: list[tuple[str, float, float]] = []

    def _apply(key: str, lo: float, hi: float):
        if key not in out:
            return
        try:
            v = out[key]
            # ints vs floats
            if isinstance(v, (int,)) and not isinstance(v, bool):
                vv = int(v)
                n = int(min(max(vv, int(lo)), int(hi)))
            else:
                vv = float(v)
                n = float(min(max(vv, float(lo)), float(hi)))
            if n != v:
                changes.append((key, v, n))
                out[key] = n
        except Exception:
            # if something is weird, leave it
            return

    # global clamps first
    for k, (lo, hi) in GLOBAL_PARAM_CLAMPS.items():
        _apply(k, lo, hi)

    # per plugin overrides
    pm = PLUGIN_PARAM_CLAMPS.get(pid, {})
    if isinstance(pm, dict):
        for k, (lo, hi) in pm.items():
            _apply(k, lo, hi)

    return out, changes
