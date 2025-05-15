# gguf_model.py
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import torch
import intel_extension_for_pytorch as ipex

class GGUFModelLoader:
    def __init__(self, model_name="mradermacher/T3Q-qwen2.5-14b-v1.0-e3-GGUF", 
                 model_file="T3Q-qwen2.5-14b-v1.0-e3.Q8_0.gguf"):
        self.device = "xpu" if torch.xpu.is_available() else "cpu"
        model_path = hf_hub_download(
            repo_id=model_name,
            filename=model_file,
            resume_download=True
        )
        
        # Intel-optimized GGUF loader
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,  # Context window
            n_gpu_layers=-1 if self.device == "xpu" else 0,  # Use all XPU layers
            n_threads=8,  # CPU threads
            n_batch=512,  # Batch size for XPU
            offload_kqv=True,  # Better memory management
            main_gpu=0 if self.device == "xpu" else None,
            verbose=False
        )
        
        # Enable Intel-specific optimizations
        if self.device == "xpu":
            self.llm.ctx = ipex.optimize(self.llm.ctx)

    def generate(self, prompt, max_tokens=200):
        output = self.llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=False
        )
        return output["choices"][0]["text"]