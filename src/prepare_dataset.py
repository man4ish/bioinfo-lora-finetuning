import json
import random
from pathlib import Path

# Paths to datasets
original_file = Path("data/bioinfo_train.jsonl")
extra_file = Path("data/bioinfo_train_extra.jsonl")

# Output files
train_file = Path("data/bioinfo_train_final.jsonl")
val_file = Path("data/bioinfo_val.jsonl")
test_file = Path("data/bioinfo_test.jsonl")

# Load datasets
def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

original_data = load_jsonl(original_file)
extra_data = load_jsonl(extra_file)

# Merge datasets
full_data = original_data + extra_data
print(f"Total examples after merge: {len(full_data)}")

# Shuffle data
random.seed(42)
random.shuffle(full_data)

# Split: 70% train, 20% val, 10% test
n = len(full_data)
train_split = int(0.7 * n)
val_split = int(0.9 * n)  # 70% train, 20% val, 10% test

train_data = full_data[:train_split]
val_data = full_data[train_split:val_split]
test_data = full_data[val_split:]

# Save to JSONL
def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")

save_jsonl(train_data, train_file)
save_jsonl(val_data, val_file)
save_jsonl(test_data, test_file)

print(f"Train examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")
print(f"Test examples: {len(test_data)}")
print("✅ Dataset preparation complete!")
