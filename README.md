# RLHF Pipeline

## Week 1 — Supervised Fine-Tuning (SFT)
- **Model**: facebook/opt-1.3b
- **Dataset**: timdettmers/openassistant-guanaco
- **Method**: LoRA + 4-bit quantization

## How to Run

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run training**:
```bash
python week1_sft/train.py
```

3. **Run inference (evaluation)**:
```bash
python week1_sft/inference.py
```

## Results

**Before SFT**:
```text
Prompt: "### Human: Explain what is machine learning\n### Assistant:"
Output: "machine learning is a type of machine learning that is used to..."
(repetitive, no structure)
```

**After SFT**:
```text
Prompt: "### Human: Explain what is machine learning\n### Assistant:"
Output: "Machine learning is a subset of AI that enables systems to learn
from data and improve over time without being explicitly programmed..."
(clear, structured, helpful)
```
