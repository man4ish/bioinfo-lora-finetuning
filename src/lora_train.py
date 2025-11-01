# src/lora_train.py
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# ------------------------------
# 1. Command-line arguments
# ------------------------------
parser = argparse.ArgumentParser(description="LoRA fine-tuning for Bioinformatics LLM")
parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
parser.add_argument("--dataset_path", type=str, default="data/bioinfo_train.jsonl")
parser.add_argument("--output_dir", type=str, default="./results/lora-adapter")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation", type=int, default=4)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
args = parser.parse_args()

print(f"Using device: {args.device}")
print(f"Model: {args.model_name}")
print(f"Dataset: {args.dataset_path}")

# ------------------------------
# 2. Load tokenizer and model
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16).to(args.device)

# ------------------------------
# 3. Configure LoRA
# ------------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# ------------------------------
# 4. Load and preprocess dataset
# ------------------------------
dataset = load_dataset("json", data_files={"train": args.dataset_path})["train"]

def format_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

def tokenize(example):
    text = format_prompt(example)
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize, batched=False)

# ------------------------------
# 5. Training setup
# ------------------------------
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation,
    num_train_epochs=args.epochs,
    logging_steps=10,
    learning_rate=args.lr,
    fp16=False,               # Set True if MPS or CUDA supports
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

# ------------------------------
# 6. Start training
# ------------------------------
print("Starting LoRA fine-tuning...")
trainer.train()

# ------------------------------
# 7. Save LoRA adapter
# ------------------------------
model.save_pretrained(args.output_dir)
print(f"✅ LoRA fine-tuning completed and saved to {args.output_dir}")
