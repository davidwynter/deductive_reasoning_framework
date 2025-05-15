import time
import torch
from deductive_ai.engine.deductive_engine import convert_nl_to_swrl


def benchmark_inference():
    torch.xpu.synchronize()
    start_time = time.time()
    
    # Warmup
    for _ in range(3):
        _ = convert_nl_to_swrl("Sample rule")
    
    # Benchmark
    torch.xpu.synchronize()
    start = time.time()
    for _ in range(10):
        _ = convert_nl_to_swrl("Sample rule")
    torch.xpu.synchronize()
    print(f"XPU Inference time: {(time.time()-start)/10:.2f}s")

# Usage in app initialization
if __name__ == "__main__":
    benchmark_inference()