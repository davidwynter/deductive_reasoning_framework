# Converts NL to SWRL
# TODO Determine best tokenizer to use
from transformers import pipeline

class SWRLGenerator:
    def __init__(self, model, tokenizer):
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200
        )
    
    def generate_swrl(self, nl_input):
        prompt = f"""Convert to SWRL:
        NL: {nl_input}
        SWRL:"""
        return self.pipe(prompt)[0]['generated_text']