import threading
import time

import pynvml

from mangajanaitrt.console import console


class VRAMMonitor:
    """Single-GPU VRAM monitor (original implementation)."""

    def __init__(self, device_id: int = 0, interval: float = 0.1) -> None:
        self.enabled = True
        self.device_id = device_id
        self.interval = interval

        self.base_used = 0
        self.peak_abs = 0
        self.peak_delta = 0

        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> None:
        if not self.enabled:
            return

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.base_used = info.used
            self.peak_abs = info.used
            self.peak_delta = 0
            pynvml.nvmlShutdown()
        except Exception as e:
            console.print(f"[yellow]VRAM monitor init failed: {e}[/]")
            self.enabled = False
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def _monitor_loop(self) -> None:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)

            while not self._stop_event.is_set():
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    used = info.used

                    self.peak_abs = max(self.peak_abs, used)

                    delta = used - self.base_used
                    self.peak_delta = max(self.peak_delta, delta)

                except Exception:
                    pass

                time.sleep(self.interval)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def stop(self):
        if not self.enabled or self._thread is None:
            return None

        self._stop_event.set()
        self._thread.join(timeout=1.0)

        return {
            "base_used_gib": self.base_used / (1024**3),
            "peak_abs_gib": self.peak_abs / (1024**3),
            "peak_delta_gib": self.peak_delta / (1024**3),
        }


class MultiGPUVRAMMonitor:
    """Multi-GPU VRAM monitor that tracks all specified devices."""

    def __init__(self, device_ids: list[int], interval: float = 0.1) -> None:
        self.enabled = True
        self.device_ids = device_ids
        self.interval = interval

        # Per-GPU stats: {device_id: {base_used, peak_abs, peak_delta}}
        self.stats: dict[int, dict[str, int]] = {}
        for dev_id in device_ids:
            self.stats[dev_id] = {
                "base_used": 0,
                "peak_abs": 0,
                "peak_delta": 0,
            }

        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self.enabled:
            return

        try:
            pynvml.nvmlInit()

            # Get baseline for all GPUs
            for dev_id in self.device_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.stats[dev_id]["base_used"] = info.used
                self.stats[dev_id]["peak_abs"] = info.used
                self.stats[dev_id]["peak_delta"] = 0

            pynvml.nvmlShutdown()
        except Exception as e:
            console.print(f"[yellow]Multi-GPU VRAM monitor init failed: {e}[/]")
            self.enabled = False
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def _monitor_loop(self) -> None:
        handles: dict[int, object] = {}

        try:
            pynvml.nvmlInit()

            # Get handles for all GPUs
            for dev_id in self.device_ids:
                handles[dev_id] = pynvml.nvmlDeviceGetHandleByIndex(dev_id)

            while not self._stop_event.is_set():
                try:
                    with self._lock:
                        for dev_id, handle in handles.items():
                            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            used = info.used

                            self.stats[dev_id]["peak_abs"] = max(
                                self.stats[dev_id]["peak_abs"], used
                            )

                            delta = used - self.stats[dev_id]["base_used"]
                            self.stats[dev_id]["peak_delta"] = max(
                                self.stats[dev_id]["peak_delta"], delta
                            )
                except Exception:
                    pass

                time.sleep(self.interval)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def stop(self) -> dict[int, dict[str, float]] | None:
        if not self.enabled or self._thread is None:
            return None

        self._stop_event.set()
        self._thread.join(timeout=1.0)

        result = {}
        with self._lock:
            for dev_id in self.device_ids:
                result[dev_id] = {
                    "base_used_gib": self.stats[dev_id]["base_used"] / (1024**3),
                    "peak_abs_gib": self.stats[dev_id]["peak_abs"] / (1024**3),
                    "peak_delta_gib": self.stats[dev_id]["peak_delta"] / (1024**3),
                }

        return result

    def get_current_usage(self) -> dict[int, float]:
        """Get current VRAM usage for all GPUs (in GiB)."""
        result = {}
        try:
            pynvml.nvmlInit()
            for dev_id in self.device_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                result[dev_id] = info.used / (1024**3)
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return result
