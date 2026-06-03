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
[paste example output from base model]
```

**After SFT**:
```text
[paste example output from fine-tuned model]
```
