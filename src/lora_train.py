# src/lora_train.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Using device: {device}")

# --- Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

# --- Configure LoRA ---
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# --- Load bioinformatics dataset ---
dataset = load_dataset("json", data_files={"train": "data/bioinfo_train.jsonl"})["train"]

def format_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

def tokenize(example):
    text = format_prompt(example)
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize)

# --- Training setup ---
training_args = TrainingArguments(
    output_dir="./results/lora-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

print("🚀 Starting LoRA fine-tuning...")
trainer.train()

model.save_pretrained("./results/lora-adapter")
print("✅ LoRA fine-tuning completed and saved to ./results/lora-adapter")
