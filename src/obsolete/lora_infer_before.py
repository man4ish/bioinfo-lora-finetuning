# src/lora_infer_before.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model.eval()

prompt = "### Instruction:\nExplain what a FASTQ file is.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, top_p=0.9)

print("\n--- Baseline Output ---\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
