# Bioinformatics LLM Fine-Tuning with LoRA

This repository demonstrates fine-tuning a lightweight LLM (TinyLlama-1.1B) on bioinformatics instruction-response prompts using Low-Rank Adaptation (LoRA). The goal is to adapt a general instruction-tuned model to answer domain-specific questions and showcase the improvements before and after fine-tuning.

---

## Objective
- Fine-tune TinyLlama for bioinformatics instructions  
- Run locally on Apple Silicon (M4 Max, MPS backend)  
- Compare outputs before and after LoRA fine-tuning  

---

## Repository Structure

```

bioinfo-lora-finetuning-demo/
├── data/
│   └── bioinfo_train.jsonl          # Instruction-response dataset
├── src/
│   ├── lora_train.py                # LoRA fine-tuning script
│   ├── lora_infer_before.py         # Baseline inference
│   ├── lora_infer_after.py          # Inference using LoRA adapter
│   └── merge_lora.py                # Merge adapter into base model
├── results/
│   ├── sample_outputs_before.txt
│   ├── sample_outputs_after.txt
│   └── merged-model/
├── requirements.txt
└── README.md

````

---

## Environment Setup

1. Clone the repository and create a virtual environment:
```bash
git clone <your-repo-url>
cd bioinfo-lora-finetuning-demo
python -m venv venv
source venv/bin/activate
````

2. Install dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Note: On macOS M4, if you encounter `sentencepiece` build errors, use:

```bash
pip install sentencepiece --prefer-binary
```

---

## Dataset

* `data/bioinfo_train.jsonl` contains bioinformatics instruction-response pairs in JSON Lines format:

```json
{"instruction": "Explain what a FASTQ file is.", "output": "A FASTQ file stores sequencing reads with quality scores..."}
{"instruction": "What is SNP annotation?", "output": "SNP annotation links single-nucleotide polymorphisms to genes and predicts functional impacts."}
```

* You can expand this dataset to hundreds of examples for improved results.

---

## Fine-Tuning

```bash
python src/lora_train.py
```

* Saves LoRA adapter to `./results/lora-adapter`
* Uses MPS GPU if available
* Training logs show decreasing loss over steps

---

## Inference

### Baseline (Before Fine-Tuning)

```bash
python src/lora_infer_before.py
```

### Fine-Tuned Model (After LoRA)

```bash
python src/lora_infer_after.py
```

Example comparison:

| Prompt         | Baseline                            | LoRA Fine-Tuned                                                                                                     |
| -------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Explain FASTQ  | “A text file containing sequences.” | “A FASTQ file stores sequencing reads with quality scores, used in genome sequencing and bioinformatics pipelines.” |
| SNP annotation | “A process in genomics.”            | “SNP annotation links single-nucleotide polymorphisms to genes and predicts functional impact.”                     |

---

## Merge LoRA Adapter

To create a standalone fine-tuned model:

```bash
python src/merge_lora.py
```

* Output: `./results/merged-model/`
* Load without PEFT adapters:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./results/merged-model").to("mps")
tokenizer = AutoTokenizer.from_pretrained("./results/merged-model")
```

---

## Results

* Train time: ~1 min per epoch on M4 Max
* Loss decrease: ~10 → 0.3
* Output improvement: Domain-specific answers with bioinformatics terms

