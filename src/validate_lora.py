# src/validate_lora.py
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from peft import PeftModel

# ------------------------------
# Load fine-tuned LoRA adapter
# ------------------------------
model_base = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(model_base, "results/lora-adapter")

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token

# ------------------------------
# Load validation/test dataset
# ------------------------------
val_dataset = load_dataset("json", data_files={"validation": "data/bioinfo_val.jsonl"})["validation"]

# ------------------------------
# Tokenize dataset for evaluation
# ------------------------------
def format_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

def tokenize(example):
    text = format_prompt(example)
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_val = val_dataset.map(tokenize, batched=False)

# ------------------------------
# Evaluate validation loss
# ------------------------------
trainer = Trainer(model=model, eval_dataset=tokenized_val, tokenizer=tokenizer)
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# ------------------------------
# Generate predictions for each instruction
# ------------------------------
results = []
for example in val_dataset:
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_length=512)
    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    results.append({
        "instruction": example['instruction'],
        "prediction": pred,
        "target": example['output']
    })

# ------------------------------
# Save predictions to JSON
# ------------------------------
import pathlib
pathlib.Path("results").mkdir(exist_ok=True)
with open("results/val_predictions.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Predictions saved to results/val_predictions.json")
