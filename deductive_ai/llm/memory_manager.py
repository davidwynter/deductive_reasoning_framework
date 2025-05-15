# memory_manager.py
import torch
import gc
from time import time
from functools import wraps

class XPUMemoryManager:
    def __init__(self, high_water_mark=0.8):
        self.high_water_mark = high_water_mark  # 80% VRAM usage threshold
        
    def _check_memory(self):
        if not torch.xpu.is_available():
            return
            
        allocated = torch.xpu.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.xpu.memory_reserved() / (1024 ** 3)
        total = torch.xpu.get_device_properties(0).total_memory / (1024 ** 3)
        
        print(f"[Memory] Used: {allocated:.2f}GB/{total:.2f}GB | Reserved: {reserved:.2f}GB")
        
        if allocated > total * self.high_water_mark:
            self.clear_cache()

    def clear_cache(self):
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.reset_peak_memory_stats()
            gc.collect()
            print("XPU cache cleared")

    def memory_monitor(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_mem = torch.xpu.memory_allocated() if torch.xpu.is_available() else 0
            start_time = time()
            
            self._check_memory()
            result = func(*args, **kwargs)
            
            duration = time() - start_time
            end_mem = torch.xpu.memory_allocated() if torch.xpu.is_available() else 0
            delta_mem = (end_mem - start_mem) / (1024 ** 2)  # MB
            
            print(f"Operation '{func.__name__}' took {duration:.2f}s, Memory Δ: {delta_mem:+.2f}MB")
            self._check_memory()
            
            return result
        return wrapper