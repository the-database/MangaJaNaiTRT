import threading
import time

import pynvml

from mangajanaitrt.console import console


class VRAMMonitor:
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
