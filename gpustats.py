"""GPU memory + timing helpers.

Uses NVML (nvidia-ml-py), which reads the WHOLE device. That matters because it
captures everything resident on the GPU regardless of backend or process -- so
the same numbers are comparable across the HF and vLLM runs.
"""
import time
import threading

try:
    import pynvml
    pynvml.nvmlInit()
    _HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML = True
except Exception:
    _NVML = False


def vram_used_mb() -> float:
    if not _NVML:
        return float("nan")
    return pynvml.nvmlDeviceGetMemoryInfo(_HANDLE).used / (1024 ** 2)


def vram_total_mb() -> float:
    if not _NVML:
        return float("nan")
    return pynvml.nvmlDeviceGetMemoryInfo(_HANDLE).total / (1024 ** 2)


class PeakVRAM:
    """Background sampler that records peak device VRAM during a `with` block."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, vram_used_mb())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = vram_used_mb()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0
