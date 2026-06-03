import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import config

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.7, 
            do_sample=True, 
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
        
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    prompts = [
        "### Human: Explain what is machine learning\n### Assistant:",
        "### Human: Write a Python function to reverse a string\n### Assistant:",
        "### Human: What is the capital of France?\n### Assistant:"
    ]

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Base Model ({config.MODEL_NAME})...")
    # Using float16 to save memory, matching standard inference patterns
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print("\n" + "="*50)
    print("--- Base Model Responses ---")
    print("="*50)
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 20)
        print("Response:\n" + generate_response(base_model, tokenizer, prompt))

    print(f"\nLoading Fine-tuned Adapter from {config.OUTPUT_DIR}...")
    try:
        ft_model = PeftModel.from_pretrained(base_model, config.OUTPUT_DIR)
    except Exception as e:
        print(f"Failed to load fine-tuned model. Ensure you have run train.py and '{config.OUTPUT_DIR}' exists.")
        print(f"Error: {e}")
        return

    print("\n" + "="*50)
    print("--- Fine-Tuned Model Responses ---")
    print("="*50)
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 20)
        print("Response:\n" + generate_response(ft_model, tokenizer, prompt))

if __name__ == "__main__":
    main()
