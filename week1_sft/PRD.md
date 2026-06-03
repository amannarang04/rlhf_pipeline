# Product Requirements Document (PRD): Week 1 SFT

## 1. Overview
The "Week 1 SFT" module represents the first stage in a Reinforcement Learning from Human Feedback (RLHF) pipeline. The core objective is to perform Supervised Fine-Tuning (SFT) on a pre-trained base language model to adapt its behavior into a conversational assistant. 

## 2. Goals & Objectives
- **Transform Base Model Behavior**: Convert a general text-completion model into an instruction-following assistant.
- **Resource Efficiency**: Ensure the fine-tuning process can be executed on consumer-grade or limited-memory GPUs without compromising the model's core capabilities.
- **Provide a Baseline**: Establish a fine-tuned baseline model that can be subsequently used for Reward Modeling (Week 2) and PPO (Week 3) in the broader RLHF pipeline.

## 3. Target Audience
- Machine Learning Engineers and Researchers building instruction-tuned Large Language Models (LLMs).
- Developers integrating conversational AI capabilities into applications.

## 4. Key Features
- **Assistant-Style Responses**: The model will learn to respond to `### Human:` prompts with appropriate `### Assistant:` completions.
- **Side-by-Side Evaluation**: The module provides built-in capabilities to directly compare the generative outputs of the original base model against the fine-tuned adapter.
- **Modular Configuration**: All training hyperparameters and model selections are centralized for easy experimentation.

## 5. User Stories
- *As an ML Engineer*, I want to fine-tune `facebook/opt-1.3b` on the `openassistant-guanaco` dataset so that the model can understand and respond to human instructions.
- *As a researcher*, I want to use PEFT (Parameter-Efficient Fine-Tuning) and 4-bit quantization so that I can train the model on a single GPU.
- *As a developer*, I want to run a quick inference script to visually compare the base model's completion against the new conversational capabilities of the fine-tuned model.

## 6. Out of Scope (For Week 1)
- Reward Modeling (RM)
- Proximal Policy Optimization (PPO)
- Full-parameter fine-tuning (only LoRA adapters will be trained).
