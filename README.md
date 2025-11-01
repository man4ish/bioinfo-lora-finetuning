
# Bioinformatics LLM Fine-Tuning with LoRA

This repository demonstrates fine-tuning a lightweight LLM (TinyLlama-1.1B) on bioinformatics instruction-response prompts using Low-Rank Adaptation (LoRA). The goal is to adapt a general instruction-tuned model to answer domain-specific questions and showcase improvements before and after fine-tuning.

---

## Objective

* Fine-tune TinyLlama for bioinformatics instructions
* Run locally on Apple Silicon (M4 Max, MPS backend)
* Compare outputs before and after LoRA fine-tuning
* Evaluate on validation/test sets and save predictions for portfolio demos

---

## Repository Structure

```
bioinfo-lora-finetuning-demo/
├── data/
│   ├── bioinfo_train_final.jsonl     # Instruction-response training dataset
│   └── bioinfo_val.jsonl             # Validation dataset
├── src/
│   ├── prepare_dataset.py            # Preprocess and create dataset splits
│   ├── lora_train.py                 # LoRA fine-tuning script
│   └── validate_lora.py              # Evaluate & save predictions
├── results/
│   ├── val_predictions.json          # Validation predictions (LoRA)
│   └── lora-adapter/                 # Saved LoRA adapter
├── requirements.txt
└── README.md
```

---

## Environment Setup

1. Clone the repository and create a virtual environment:

```bash
git clone <your-repo-url>
cd bioinfo-lora-finetuning-demo
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> Note: On macOS M4, if you encounter `sentencepiece` build errors, use:

```bash
pip install sentencepiece --prefer-binary
```

---

## Dataset

* `data/bioinfo_train_final.jsonl` contains bioinformatics instruction-response pairs in JSON Lines format.
* `data/bioinfo_val.jsonl` is used for validation.

Example JSON line:

```json
{"instruction": "Explain what a FASTQ file is.", "output": "A FASTQ file stores sequencing reads with quality scores."}
```

---

## Fine-Tuning

```bash
python src/lora_train.py \
  --train_dataset data/bioinfo_train_final.jsonl \
  --val_dataset data/bioinfo_val.jsonl \
  --epochs 3 \
  --batch_size 2 \
  --gradient_accumulation 4 \
  --output_dir results/lora-adapter
```

* Saves LoRA adapter to `./results/lora-adapter`
* Uses MPS GPU if available
* Training logs show decreasing loss over steps

---

## Validation & Predictions

```bash
python src/validate_lora.py
```

* Computes evaluation loss on validation set
* Generates predictions for each instruction
* Saves `results/val_predictions.json` for portfolio showcase

**Example output:**

```json
{
  "instruction": "Explain what a FASTQ file is.",
  "prediction": "A FASTQ file stores sequencing reads with quality scores, used in genome sequencing...",
  "target": "A FASTQ file stores sequencing reads with quality scores."
}
```

---

## Before vs After LoRA

* You can compare your baseline model (TinyLlama) and LoRA fine-tuned outputs using the saved predictions.
* Example table:

| Instruction    | Baseline Output                     | LoRA Fine-Tuned                                                                                                     | Target                                                                                           |
| -------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Explain FASTQ  | “A text file containing sequences.” | “A FASTQ file stores sequencing reads with quality scores, used in genome sequencing and bioinformatics pipelines.” | “A FASTQ file stores sequencing reads with quality scores.”                                      |
| SNP annotation | “A process in genomics.”            | “SNP annotation links single-nucleotide polymorphisms to genes and predicts functional impact.”                     | “SNP annotation links single-nucleotide polymorphisms to genes and predicts functional impacts.” |

---

## Results

* Train time: ~1 min per epoch on M4 Max
* Loss decrease: ~10 → ~0.3
* Validation loss: ~4.44
* Output improvement: Domain-specific answers with bioinformatics terminology
* Predictions saved for portfolio: `results/val_predictions.json`
