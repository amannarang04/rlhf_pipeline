# Technical Requirements Document (TRD): Week 1 SFT

## 1. System Architecture
This module implements a Supervised Fine-Tuning (SFT) pipeline utilizing Hugging Face's `trl` (Transformer Reinforcement Learning) library. 

### 1.1 Components
- **Base Model**: `facebook/opt-1.3b` (Causal LM).
- **Dataset**: `timdettmers/openassistant-guanaco` (Instruction/Response pairs).
- **Quantization Engine**: `bitsandbytes` (NF4 4-bit precision).
- **Adapter Strategy**: PEFT LoRA (Low-Rank Adaptation).
- **Trainer**: `SFTTrainer` from the `trl` library.

## 2. Technical Specifications

### 2.1 Model Quantization (QLoRA Approach)
To fit the model into limited VRAM, the base model is loaded in 4-bit precision using `BitsAndBytesConfig`:
- `load_in_4bit`: True
- `bnb_4bit_quant_type`: "nf4" (Normalized Float 4)
- `bnb_4bit_compute_dtype`: `torch.float16`
- `bnb_4bit_use_double_quant`: True

### 2.2 LoRA Configuration
Instead of updating all model weights, we inject trainable rank decomposition matrices:
- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.05
- **Target Modules**: `["q_proj", "v_proj"]`
- **Task Type**: `CAUSAL_LM`
- **Bias**: "none"

### 2.3 Training Hyperparameters
Configured via `SFTConfig` and centralized in `config.py`:
- **Sequence Length**: 512 tokens
- **Epochs**: 3
- **Batch Size**: 4 (Per Device)
- **Gradient Accumulation Steps**: 4 (Effective batch size = 16)
- **Learning Rate**: 2e-4
- **LR Scheduler**: Cosine with a 3% (`0.03`) warmup ratio
- **Optimizer**: Default AdamW (via Trainer)

### 2.4 Data Processing (`utils.py`)
The `format_prompt` function standardizes the dataset into the expected causal modeling format:
```text
### Human: {instruction}
### Assistant: {response}<|endoftext|>
```

## 3. Project Structure
- `config.py`: Centralized configuration variables.
- `utils.py`: Helper functions for dataset formatting.
- `train.py`: The main execution script that loads the model, applies LoRA, formats the dataset, and runs the `SFTTrainer`.
- `inference.py`: Evaluation script that loads both the base model and the trained LoRA adapters (`PeftModel`) to generate side-by-side completions for visual inspection.
- `requirements.txt`: Python dependencies.

## 4. Hardware & Software Requirements
- **Hardware**: Minimum 1x NVIDIA GPU (8GB+ VRAM recommended due to 4-bit quantization).
- **Environment**: Python 3.8+
- **Dependencies**: `torch`, `transformers`, `trl`, `datasets`, `peft`, `bitsandbytes`, `accelerate`.

## 5. Artifacts & Outputs
Upon successful training, the script will output the LoRA adapter weights, configuration, and tokenizer files to the directory specified by `OUTPUT_DIR` (default: `./sft_output`). Note: The base model weights are *not* saved, only the adapter weights.
