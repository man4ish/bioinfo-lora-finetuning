# src/lora_infer_after.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./results/lora-adapter"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter_path)
model.to(device)
model.eval()

prompt = "### Instruction:\nExplain what a FASTQ file is.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, top_p=0.9)

print("\n--- Fine-Tuned Output ---\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
