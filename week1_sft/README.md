# Week 1: Supervised Fine-Tuning (SFT)

This repository contains the first stage of an RLHF (Reinforcement Learning from Human Feedback) pipeline: **Supervised Fine-Tuning (SFT)**. 

Using the `trl` and `peft` libraries, we fine-tune a base causal language model (`facebook/opt-1.3b`) on an instruction-following dataset (`openassistant-guanaco`) to create a conversational assistant. We employ 4-bit quantization (QLoRA) to allow training on consumer-grade hardware.

## Project Structure
- `config.py`: Hyperparameters, model paths, and dataset settings.
- `train.py`: Main script to execute the SFT training loop.
- `inference.py`: Script to compare the generated responses of the base model vs. the fine-tuned model.
- `utils.py`: Utility functions, including prompt formatting.
- `requirements.txt`: Required python packages.

## Installation

Ensure you have Python 3.8+ and a CUDA-capable GPU. Install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Note: `bitsandbytes` requires a compatible CUDA runtime).*

## Configuration

All training parameters are managed in `config.py`. By default, the script is configured to use:
- **Base Model**: `facebook/opt-1.3b`
- **Dataset**: `timdettmers/openassistant-guanaco`
- **Output Directory**: `./sft_output`

If you wish to change the model, dataset, or training hyperparameters (like learning rate, batch size, or LoRA settings), modify `config.py` before running the scripts.

## Usage

### 1. Training

Run the `train.py` script to begin fine-tuning. The script will download the model, apply 4-bit quantization, inject LoRA adapters, and begin the training loop.

```bash
python train.py
```

Upon completion, the LoRA adapter weights and tokenizer will be saved to the `./sft_output` directory.

### 2. Inference & Evaluation

To evaluate the success of the fine-tuning process, run the inference script. This script loads the base model and then loads the trained LoRA adapter on top of it. It will prompt both models with identical questions so you can directly compare their responses.

```bash
python inference.py
```

## Documentation
For more detailed information regarding the design and architecture of this module, refer to:
- [Product Requirements Document (PRD.md)](./PRD.md)
- [Technical Requirements Document (TRD.md)](./TRD.md)
