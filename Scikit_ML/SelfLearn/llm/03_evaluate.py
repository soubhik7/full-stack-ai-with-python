import os
from transformers import AutoTokenizer, GPT2LMHeadModel

# Step 6: Evaluate and Test Your LLM 🧪

def clean_code(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    clean_lines = [line for line in lines if not line.strip().startswith('#')]
    return ''.join(clean_lines)

if __name__ == "__main__":
    model_dir = "./fine_tuned_model"
    
    # We check if the fine-tuned model exists, otherwise we fallback to base GPT-2 for testing
    if not os.path.exists(model_dir):
        print(f"Warning: Model directory '{model_dir}' not found. Using pre-trained 'gpt2' instead.")
        print("To test your fine-tuned model, make sure you run 'python 02_train.py' first.")
        model_name_or_path = "gpt2"
    else:
        model_name_or_path = model_dir
    
    print(f"\nLoading model from '{model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    
    # Ask the user for input from the terminal
    print("\n" + "="*50)
    prompt = input("Enter your Python-related prompt or code (or 'quit' to exit):\n> ")
    if prompt.lower() in ['quit', 'exit']:
        print("Exiting.")
        exit()
    
    print("\nTokenizing input...")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("\nGenerating response from the model...")
    # Setting pad_token_id to avoid console warnings during generation
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=150, 
        pad_token_id=tokenizer.eos_token_id, 
        do_sample=True,      # Introduce some generation diversity
        temperature=0.7,     # High temperature = more random, Low = more deterministic
        top_k=50
    )
    
    # Decode and print the result
    print("\n--- GENERATED RESPONSE ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("--------------------------\n")
