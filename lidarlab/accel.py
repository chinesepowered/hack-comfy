"""CPU acceleration helpers.

Modern Xeons (Sapphire Rapids and newer: AMX / AVX512-BF16) run bf16 matmuls
~5x faster than fp32, but ComfyUI keeps CPU inference in fp32. These helpers
detect such CPUs so LiDAR Lab can autocast its heavy model calls to bf16.
"""
from __future__ import annotations

import functools
import os


@functools.lru_cache(maxsize=1)
def cpu_bf16_ok() -> bool:
    if os.environ.get("LIDARLAB_CPU_BF16", "auto").lower() in ("0", "off", "false"):
        return False
    if os.environ.get("LIDARLAB_CPU_BF16", "auto").lower() in ("1", "on", "true"):
        return True
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
        return ("amx_bf16" in flags) or ("avx512_bf16" in flags)
    except OSError:
        return False
