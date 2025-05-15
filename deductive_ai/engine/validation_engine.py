import re
import torch
from typing import Dict, List
from deductive_ai.llm.memory_manager import XPUMemoryManager

class ValidationEngine:
    def __init__(self, converter=None, ontology_manager=None, tokenizer=None, model=None):
        """
        Unified validation engine with three validation modes:
        1. NL-to-SWRL conversion validation
        2. Ontology-aware SWRL validation 
        3. XPU-optimized model validation
        """
        self.mem_manager = XPUMemoryManager()
        self.converter = converter
        self.om = ontology_manager
        self.tokenizer = tokenizer
        self.model = model

    def validate_rule(self, nl_rule: str) -> Dict:
        """Top-level validation combining all checks"""
        self.mem_manager.clear_cache()
        
        try:
            with torch.xpu.stream(torch.xpu.Stream()):
                # Convert NL to SWRL if converter exists
                swrl_rule = self.converter.convert(nl_rule) if self.converter else nl_rule
                
                # Run all available validations
                results = {
                    "swrl": swrl_rule,
                    "syntax_valid": self._check_swrl_syntax(swrl_rule),
                    "ontology_errors": self.validate_swrl(swrl_rule) if self.om else [],
                    "model_validation": self._validate_with_model(swrl_rule) if self.model else None,
                    "memory_used": f"{torch.xpu.memory_allocated()/(1024**2):.2f}MB"
                }
                
                results["overall_valid"] = (
                    results["syntax_valid"] and
                    not results["ontology_errors"] and
                    (results["model_validation"]["valid"] if self.model else True)
                )
                
            return results
            
        finally:
            self.mem_manager.clear_cache()

    # Version 1: Basic Syntax Validation
    def _check_swrl_syntax(self, rule: str) -> bool:
        """Check basic SWRL syntax structure"""
        return bool(re.search(r"\?\w+\s*->\s*\?\w+", rule))  # More robust pattern

    # Version 2: Ontology Validation
    def validate_swrl(self, swrl_rule: str) -> List[str]:
        """Check ontology term usage"""
        errors = []
        # Extract class-like patterns (improved regex)
        classes = re.findall(r"(?<=\b)[A-Z][a-zA-Z0-9]*(?=\()", swrl_rule)
        properties = re.findall(r"\b[a-z][a-zA-Z0-9]*(?=\()", swrl_rule)
        
        for cls in set(classes):  # Deduplicate
            if not self.om.validate_class(cls):
                errors.append(f"Undefined class: {cls} (Suggestions: {self.om.suggest_similar_class(cls)})")
                
        for prop in set(properties):
            if not self.om.validate_property(prop):
                errors.append(f"Undefined property: {prop}")
                
        return errors

    # Version 3: Model-based Validation (XPU optimized)
    def _validate_with_model(self, rule_text: str) -> Dict:
        """Validate using LLM with XPU acceleration"""
        with torch.xpu.amp.autocast(enabled=True):
            inputs = self.tokenizer(
                rule_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to("xpu")
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9
            )
            
            validation_result = self._parse_model_output(
                outputs.cpu().detach().numpy()
            )
            
        return {
            "valid": validation_result["valid"],
            "feedback": validation_result.get("feedback", "")
        }

    def _parse_model_output(self, outputs) -> Dict:
        """Parse model's validation feedback"""
        # Implementation depends on your model's output format
        return {"valid": True}  # Placeholder