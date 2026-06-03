import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import RewardTrainer, RewardConfig
import config

def main():
    print("Step 1 - Load dataset...")
    dataset = load_dataset(config.DATASET_NAME, split="train")
    
    # Print first sample to understand structure
    print("Dataset Sample:", dataset[0])

    print("Step 2 - Build Reward Model...")
    try:
        # Load SFT model but replace the LM head with a scalar regression head
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.SFT_MODEL_PATH,
            num_labels=1,           # outputs a single scalar score
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(config.SFT_MODEL_PATH)
    except Exception as e:
        print(f"Failed to load SFT model from {config.SFT_MODEL_PATH}. Falling back to base model 'facebook/opt-1.3b'")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "facebook/opt-1.3b",
            num_labels=1,
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    reward_model.config.pad_token_id = tokenizer.eos_token_id

    # Apply LoRA to prevent OOM and FP16 gradient unscale errors on full-parameter training
    print("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    reward_model = get_peft_model(reward_model, peft_config)
    reward_model.print_trainable_parameters()

    print("Step 3 - Format data into pairs...")
    def format_pairs(sample):
        return {
            "input_ids_chosen": tokenizer(
                sample["chosen"],
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH
            )["input_ids"],
            "input_ids_rejected": tokenizer(
                sample["rejected"],
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH
            )["input_ids"]
        }
    
    dataset = dataset.map(format_pairs, batched=False)

    print("Step 4 - Train with TRL's RewardTrainer...")
    training_args = RewardConfig(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = RewardTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(config.OUTPUT_DIR)
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
