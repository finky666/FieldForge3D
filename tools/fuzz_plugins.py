"""
FieldForge3D - Plugin Fuzzer (crash-safe)

WHY:
Some plugins may hard-crash the Python process (Numba / native libs).
Therefore each test case runs in a separate subprocess.

RUN:
  python tools/fuzz_plugins.py --iters 1000 --N 96 --out logs/fuzz.jsonl

LOG:
  JSONL, one record per case. Records include:
  ok / error / traceback / timeout / crash / returncode / seconds / params
"""

import argparse
import importlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

DEFAULT_N = 96
DEFAULT_BOUNDS = 1.5
DEFAULT_ISO = 0.6

MAX_N = 160
MAX_SECONDS_PER_CASE = 45.0

HARD_CLAMPS = {
    "STEPS": (1000, 80000),
    "MAX_ITER": (1, 300),
    "ITER": (1, 300),
}

# Optional: if you want to fuzz string enums, fill this
ENUM_CHOICES: Dict[str, list] = {}


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class ParamSpec:
    name: str
    vtype: str  # "float" | "int" | "bool" | "str" | "obj"
    default: Any
    lo: Optional[float] = None
    hi: Optional[float] = None


def infer_specs_from_defaults(defaults: Dict[str, Any]) -> Dict[str, ParamSpec]:
    specs: Dict[str, ParamSpec] = {}
    for k, v in defaults.items():
        if v is None or callable(v) or isinstance(v, (dict, list, tuple, set)):
            specs[k] = ParamSpec(k, "obj", v)
            continue
        if isinstance(v, str):
            specs[k] = ParamSpec(k, "str", v)
            continue
        if isinstance(v, bool):
            specs[k] = ParamSpec(k, "bool", v)
            continue
        if isinstance(v, int) and not isinstance(v, bool):
            d = int(v)
            span = max(5, int(abs(d) * 0.5))
            lo = d - span
            hi = d + span
            if k in HARD_CLAMPS:
                lo2, hi2 = HARD_CLAMPS[k]
                lo = max(lo, lo2)
                hi = min(hi, hi2)
            specs[k] = ParamSpec(k, "int", d, float(lo), float(hi))
            continue
        try:
            d = float(v)
        except Exception:
            specs[k] = ParamSpec(k, "obj", v)
            continue
        span = abs(d) * 0.75
        if span < 0.05:
            span = 0.25
        lo = d - span
        hi = d + span
        if d > 0 and k.upper() in ("FREQ", "GAIN", "SIGMA", "RHO", "BETA", "DT", "ISO", "SCALE"):
            lo = max(lo, 0.0001)
        specs[k] = ParamSpec(k, "float", d, lo, hi)
    return specs


def mutate_params(rng: random.Random, specs: Dict[str, ParamSpec], base: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, spec in specs.items():
        if spec.vtype == "obj":
            out[k] = base[k]
            continue
        if spec.vtype == "str":
            choices = ENUM_CHOICES.get(k)
            if choices and rng.random() < 0.25:
                out[k] = rng.choice(choices)
            else:
                out[k] = base[k]
            continue
        if spec.vtype == "bool":
            out[k] = (not bool(base[k])) if (rng.random() < 0.15) else bool(base[k])
            continue
        if spec.vtype == "int":
            lo = safe_int(spec.lo, safe_int(base[k], 0))
            hi = safe_int(spec.hi, safe_int(base[k], 0))
            if lo > hi:
                lo, hi = hi, lo
            out[k] = rng.randint(lo, hi)
            continue
        # float
        lo = safe_float(spec.lo, safe_float(base[k], 0.0))
        hi = safe_float(spec.hi, safe_float(base[k], 0.0))
        if lo > hi:
            lo, hi = hi, lo
        if rng.random() < 0.6:
            out[k] = rng.uniform(lo, hi)
        else:
            mu = safe_float(base[k], 0.0)
            sigma = (hi - lo) / 6.0 if (hi - lo) > 0 else 1.0
            out[k] = clamp(rng.gauss(mu, sigma), lo, hi)

    for kk, (lo, hi) in HARD_CLAMPS.items():
        if kk in out:
            out[kk] = int(clamp(safe_int(out[kk], lo), lo, hi))

    return out


def load_plugins(plugins_dir: Path):
    sys.path.insert(0, str(plugins_dir.parent.resolve()))
    modules = []
    for py in sorted(plugins_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        mod_name = f"plugins.{py.stem}"
        try:
            mod = importlib.import_module(mod_name)
            modules.append((py.name, mod_name, mod))
        except Exception:
            modules.append((py.name, mod_name, None))
    return modules


# -----------------------------
# WORKER (single case execution)
# -----------------------------
def worker_run_case(case_path: str) -> int:
    """
    Read case JSON, run plugin compute(params), write result JSON to stdout.
    Exit code:
      0 = ok
      2 = python exception
      3 = non-finite output
      4 = protocol error
    """
    try:
        case = json.loads(Path(case_path).read_text(encoding="utf-8"))
        plugins_dir = Path(case["plugins_dir"]).resolve()
        sys.path.insert(0, str(plugins_dir.parent.resolve()))

        module_name = case["module"]
        params = case["params"]

        mod = importlib.import_module(module_name)

        if not hasattr(mod, "compute"):
            raise AttributeError("Plugin has no compute()")

        t0 = time.time()
        try:
            res = mod.compute(params)
        except TypeError:
            # support alternative keyword forms just in case
            try:
                res = mod.compute(params=params)
            except TypeError:
                res = mod.compute(settings=params)

        # normalize output
        arr = None
        if isinstance(res, np.ndarray):
            arr = res
        elif isinstance(res, dict):
            for key in ("field", "grid", "volume", "data"):
                if key in res and isinstance(res[key], np.ndarray):
                    arr = res[key]
                    break
        if arr is None:
            raise TypeError(f"Unexpected compute() result type: {type(res)}")

        secs = time.time() - t0
        if not np.isfinite(arr).all():
            out = {"ok": False, "naninf": True, "seconds": secs}
            print(json.dumps(out, ensure_ascii=False))
            return 3

        out = {"ok": True, "naninf": False, "seconds": secs}
        print(json.dumps(out, ensure_ascii=False))
        return 0

    except Exception as e:
        out = {
            "ok": False,
            "error": repr(e),
            "traceback": traceback.format_exc(limit=80),
        }
        print(json.dumps(out, ensure_ascii=False))
        return 2


# -----------------------------
# CONTROLLER (spawns workers)
# -----------------------------
def main_controller(args):
    N = int(clamp(args.N, 16, MAX_N))
    bounds = float(args.bounds)
    iso = float(args.iso)
    timeout = float(args.timeout)

    plugins_dir = Path(args.plugins).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    modules = load_plugins(plugins_dir)
    loaded = [(fn, mn, m) for (fn, mn, m) in modules if m is not None]
    failed_import = [(fn, mn) for (fn, mn, m) in modules if m is None]

    print(f"[FUZZ] plugins_dir={plugins_dir}")
    print(f"[FUZZ] loaded={len(loaded)} failed_import={len(failed_import)} N={N} iters={args.iters} seed={args.seed}")
    if failed_import:
        print("[FUZZ] Import failures:")
        for fn, mn in failed_import:
            print(f"  - {fn} ({mn})")

    tmp_dir = out_path.parent / "_tmp_cases"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "type": "header",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "N": N,
            "bounds": bounds,
            "iso": iso,
            "iters": args.iters,
            "seed": args.seed,
            "timeout": timeout,
            "loaded": len(loaded),
            "failed_import": failed_import,
        }, ensure_ascii=False) + "\n")

        for i in range(args.iters):
            plugin_file, module_name, mod = loaded[i % len(loaded)]

            defaults: Dict[str, Any] = {}
            if hasattr(mod, "get_defaults"):
                try:
                    defaults = dict(mod.get_defaults())
                except Exception:
                    defaults = {}

            specs = infer_specs_from_defaults(defaults)
            params = mutate_params(rng, specs, defaults)

            # enforce global bounds (don’t let plugins escape)
            params["N"] = N
            params["BOUNDS"] = bounds
            if "ISO" in params:
                params["ISO"] = iso

            case = {
                "plugins_dir": str(plugins_dir),
                "module": module_name,
                "params": params,
            }
            case_path = tmp_dir / f"case_{i:06d}.json"
            case_path.write_text(json.dumps(case, ensure_ascii=False, default=str), encoding="utf-8")

            start = time.time()
            ok = False
            naninf = False
            crash = False
            timeout_hit = False
            rc = None
            err = None
            tb = None
            secs = None

            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--case", str(case_path)]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                rc = p.returncode
                secs = time.time() - start

                # If the process crashed hard, stdout may be empty
                if p.stdout.strip():
                    result = json.loads(p.stdout.strip().splitlines()[-1])
                    ok = bool(result.get("ok", False))
                    naninf = bool(result.get("naninf", False))
                    # worker's seconds are inside plugin time; controller secs include spawn overhead
                    if not ok:
                        err = result.get("error")
                        tb = result.get("traceback")
                else:
                    # no output -> treat as crash
                    crash = True
                    ok = False

                if rc not in (0, 2, 3, 4) and not ok:
                    crash = True

                if rc == 3:
                    naninf = True

            except subprocess.TimeoutExpired:
                timeout_hit = True
                ok = False
                secs = time.time() - start
                err = "TimeoutExpired"

            rec = {
                "type": "case",
                "i": i,
                "plugin_file": plugin_file,
                "module": module_name,
                "ok": ok,
                "naninf": naninf,
                "timeout": timeout_hit,
                "crash": crash,
                "returncode": rc,
                "seconds": secs,
                "params": params,
            }
            if not ok:
                rec["error"] = err
                if tb:
                    rec["traceback"] = tb

            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

            if (i + 1) % 25 == 0:
                print(f"[FUZZ] {i+1}/{args.iters} ...")

    print(f"[FUZZ] done -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--N", type=int, default=DEFAULT_N)
    ap.add_argument("--bounds", type=float, default=DEFAULT_BOUNDS)
    ap.add_argument("--iso", type=float, default=DEFAULT_ISO)
    ap.add_argument("--timeout", type=float, default=MAX_SECONDS_PER_CASE)
    ap.add_argument("--out", type=str, default="logs/fuzz.jsonl")
    ap.add_argument("--plugins", type=str, default="plugins")

    # worker mode
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--case", type=str, default="")

    args = ap.parse_args()

    if args.worker:
        if not args.case:
            print(json.dumps({"ok": False, "error": "Missing --case"}, ensure_ascii=False))
            raise SystemExit(4)
        raise SystemExit(worker_run_case(args.case))

    main_controller(args)


if __name__ == "__main__":
    main()
