def format_prompt(example):
    """
    Formats the dataset into a prompt-response format.
    Handles different possible column names (input/output, instruction/response).
    """
    # If it already has a 'text' column and no 'input'/'output', it might already be formatted
    if 'text' in example and 'input' not in example and 'instruction' not in example:
        return {"text": example["text"]}
        
    input_text = example.get('input', example.get('instruction', ''))
    output_text = example.get('output', example.get('response', ''))
    
    return {"text": f"### Human: {input_text}\n### Assistant: {output_text}<|endoftext|>"}
