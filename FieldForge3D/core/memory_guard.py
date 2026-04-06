# core/memory_guard.py
from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Optional, Dict

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


@dataclass
class MemInfo:
    total_bytes: int
    avail_bytes: int
    used_bytes: int
    source: str


@dataclass
class GuardEstimate:
    N: int
    voxels: int
    est_bytes: int
    est_gb: float
    avail_gb: float
    ratio: float  # est/avail
    level: str    # green/yellow/red
    note: str
    breakdown: Dict[str, int]


def _bytes_to_gb(b: int) -> float:
    return b / (1024 ** 3)


def get_mem_info() -> MemInfo:
    # Best: psutil
    if psutil is not None:
        vm = psutil.virtual_memory()
        return MemInfo(int(vm.total), int(vm.available), int(vm.used), "psutil")

    # Linux fallback
    if sys.platform.startswith("linux"):
        try:
            total = 0
            avail = 0
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024
            if total > 0 and avail > 0:
                used = max(0, total - avail)
                return MemInfo(total, avail, used, "procfs")
        except Exception:
            pass

    # Worst-case fallback (conservative)
    total = 8 * 1024 ** 3
    avail = 2 * 1024 ** 3
    used = total - avail
    return MemInfo(total, avail, used, "fallback")


def estimate_memory_bytes(
    N: int,
    *,
    dtype_field: str = "float32",
    extra_buffers: Optional[Dict[str, str]] = None,
    overhead_factor: float = 1.20,
) -> Dict[str, int]:
    vox = int(N) * int(N) * int(N)

    dtype_sizes = {
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "int32": 4,
        "uint32": 4,
        "float32": 4,
        "float64": 8,
    }

    def sz(dtype: str) -> int:
        return int(dtype_sizes.get(dtype, 4))

    breakdown: Dict[str, int] = {}

    breakdown["field"] = vox * sz(dtype_field)
    breakdown["mask_u8"] = vox * sz("uint8")
    # conservative: gradients (some plugins + VTK pipelines allocate big temps)
    breakdown["grad_xyz_f32"] = vox * sz("float32") * 3

    if extra_buffers:
        for k, dt in extra_buffers.items():
            breakdown[str(k)] = vox * sz(str(dt))

    base = int(sum(breakdown.values()))
    breakdown["overhead"] = int(base * max(0.0, float(overhead_factor) - 1.0))
    breakdown["total"] = base + breakdown["overhead"]
    return breakdown


def guard(
    N: int,
    *,
    extra_buffers: Optional[Dict[str, str]] = None,
    overhead_factor: float = 1.20,
    yellow_ratio: float = 0.45,
    red_ratio: float = 0.70,
) -> GuardEstimate:
    mi = get_mem_info()
    breakdown = estimate_memory_bytes(N, extra_buffers=extra_buffers, overhead_factor=overhead_factor)
    est = int(breakdown["total"])
    avail = max(1, int(mi.avail_bytes))

    ratio = est / avail
    level = "green"
    note = "OK"

    if ratio >= red_ratio:
        level = "red"
        note = "High crash risk (MemoryError / VTK allocation)."
    elif ratio >= yellow_ratio:
        level = "yellow"
        note = "Borderline. Might work, might crash depending on plugin + VTK temps."

    return GuardEstimate(
        N=int(N),
        voxels=int(N) * int(N) * int(N),
        est_bytes=est,
        est_gb=_bytes_to_gb(est),
        avail_gb=_bytes_to_gb(mi.avail_bytes),
        ratio=float(ratio),
        level=level,
        note=f"{note} (mem source: {mi.source})",
        breakdown={k: int(v) for k, v in breakdown.items()},
    )


def suggest_lower_N(N: int, target_ratio: float, current_ratio: float) -> int:
    # est ~ N^3 => newN = N * (target/current)^(1/3)
    if current_ratio <= 0:
        return int(N)
    scale = (float(target_ratio) / float(current_ratio)) ** (1.0 / 3.0)
    newN = int(math.floor(int(N) * scale))
    # snap down to multiple of 8
    newN = max(16, (newN // 8) * 8)
    if newN >= int(N):
        newN = max(16, ((int(N) - 1) // 8) * 8)
    return int(newN)
