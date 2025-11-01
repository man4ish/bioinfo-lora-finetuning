# src/merge_lora.py
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter = "./results/lora-adapter"
merged_output = "./results/merged-model"

model = AutoModelForCausalLM.from_pretrained(base)
lora_model = PeftModel.from_pretrained(model, adapter)
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained(merged_output)

print(f"✅ Merged model saved to {merged_output}")
