import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import config

def main():
    print("Loading validation dataset...")
    # Taking a subset of 100 samples from the validation set for quick evaluation
    val_dataset = load_dataset(config.DATASET_NAME, split="test").select(range(100))
    
    print(f"Loading trained reward model from {config.OUTPUT_DIR}...")
    try:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.OUTPUT_DIR,
            num_labels=1,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR)
    except Exception as e:
        print(f"Error loading model from {config.OUTPUT_DIR}. Make sure train_reward_model.py ran successfully.")
        print(f"Error details: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    reward_model.config.pad_token_id = tokenizer.eos_token_id

    print("Evaluating pairwise accuracy...")
    correct = 0
    total = 0

    reward_model.eval()
    with torch.no_grad():
        for sample in val_dataset:
            # Tokenize chosen and rejected
            chosen_tokens = tokenizer(
                sample["chosen"], 
                return_tensors="pt", 
                truncation=True, 
                max_length=config.MAX_SEQ_LENGTH
            )
            rejected_tokens = tokenizer(
                sample["rejected"], 
                return_tensors="pt", 
                truncation=True, 
                max_length=config.MAX_SEQ_LENGTH
            )
            
            # Move to device
            chosen_input_ids = chosen_tokens["input_ids"].to(reward_model.device)
            chosen_attention_mask = chosen_tokens["attention_mask"].to(reward_model.device)
            
            rejected_input_ids = rejected_tokens["input_ids"].to(reward_model.device)
            rejected_attention_mask = rejected_tokens["attention_mask"].to(reward_model.device)
            
            # Get scores
            score_chosen = reward_model(
                input_ids=chosen_input_ids, 
                attention_mask=chosen_attention_mask
            ).logits[0].item()
            
            score_rejected = reward_model(
                input_ids=rejected_input_ids, 
                attention_mask=rejected_attention_mask
            ).logits[0].item()

            if score_chosen > score_rejected:
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print(f"Reward Model Accuracy: {accuracy:.2f}%")
    
    if accuracy >= 70:
        print("Result: Great result!")
    elif accuracy >= 60:
        print("Result: Good result.")
    else:
        print("Result: Model might not be learning effectively (accuracy < 60%). Consider lowering learning rate to 5e-6.")

if __name__ == "__main__":
    main()
