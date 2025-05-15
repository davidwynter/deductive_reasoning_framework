from deductive_ai.llm.memory_manager import XPUMemoryManager
from transformers import pipeline
from deductive_reasoning_framework.deductive_ai.core.ontology_manager import OntologyManager
import torch
from deductive_ai.llm.gguf_model_loader import GGUFModelLoader

class NLToSWRLConverter:
    def __init__(self, ontology_path, model_name="meta-llama/Llama-2-13b-chat-hf"):
        self.ontology = OntologyManager(ontology_path)
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.mem_manager = XPUMemoryManager()
        
    @XPUMemoryManager.memory_monitor
    def convert(self, nl_rule: str) -> str:
        prompt = f"""Convert this natural language rule to SWRL:
        Ontology Terms: {self.ontology.list_classes()} | {self.ontology.list_properties()}
        Input: {nl_rule}
        SWRL:"""
        
        try:
            result = self.pipe(
                prompt,
                max_new_tokens=200,
                return_full_text=False
            )
            
            return self._postprocess_swrl(result[0]['generated_text'])
        finally:
            self.mem_manager.clear_cache()
    
    def _postprocess_swrl(self, raw_swrl: str) -> str:
        # Add validation and formatting logic
        return raw_swrl.strip().replace("\n", " ")
    

# Support binaries in GGUF format
class NLToSWRLGGUFConverter:
    def __init__(self, ontology_path):
        self.ontology = OntologyManager(ontology_path)
        self.model = GGUFModelLoader()
        self.mem_manager = XPUMemoryManager()
        
    @XPUMemoryManager.memory_monitor
    def convert(self, nl_rule: str) -> str:
        system_prompt = """You are an ontology expert converting natural language to SWRL rules.
        Available classes: {classes}
        Available properties: {properties}
        Always use valid ontology terms!"""
        
        prompt = f"""<|im_start|>system
        {system_prompt.format(
            classes=", ".join(self.ontology.list_classes()),
            properties=", ".join(self.ontology.list_properties())
        )}
        <|im_end|>
        <|im_start|>user
        Convert to SWRL: {nl_rule}<|im_end|>
        <|im_start|>assistant
        SWRL: """
        
        try:
            return self._clean_swrl_output(
                self.model.generate(prompt)
            )
        finally:
            self.mem_manager.clear_cache()
            
    def _clean_swrl_output(self, raw: str) -> str:
        # Remove any chat formatting artifacts
        return raw.split("<|im_end|>")[0].strip()