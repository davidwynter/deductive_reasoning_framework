# Check memory usage of the GPU
import threading
import time
from deductive_ai.llm.memory_manager import XPUMemoryManager

class MemoryMonitor:
    def __init__(self, interval=30):
        self.manager = XPUMemoryManager()
        self.interval = interval
        self._stop_event = threading.Event()

    def start(self):
        def monitor_loop():
            while not self._stop_event.is_set():
                self.manager._check_memory()
                time.sleep(self.interval)
                
        threading.Thread(target=monitor_loop, daemon=True).start()

    def stop(self):
        self._stop_event.set()