# llm_train.py
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from ctransformers import AutoModelForCausalLM


def train_llm(use_model="Llama"):
    if use_model == "Llama":
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", 
                                            load_in_4bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained("mradermacher/T3Q-qwen2.5-14b-v1.0-e3-GGUF", 
                                            model_file="T3Q-qwen2.5-14b-v1.0-e3.Q8_0.gguf",
                                            gpu_layers=-1)

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        output_dir="./llama-swrl"
    )

# Train with NL-SWRL pairs: {"text": "If <NL rule> → <SWRL>"}