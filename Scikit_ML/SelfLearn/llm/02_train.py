import os
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Step 4: Preprocess the Data 📄
# Using Hugging Face Transformers GPT-2 as an example.
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 does not have a pad token by default, so we set it to eos_token
tokenizer.pad_token = tokenizer.eos_token

def prepare_dataset(file_path):
    """
    Reads the cleaned text file and tokenizes the text into a Hugging Face Dataset.
    This fixes the 'tokenized_code' format to align with the Trainer API.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
    # Tokenize the entire text in chunks 
    tokenized_output = tokenizer(
        text, 
        truncation=True, 
        max_length=512, 
        return_overflowing_tokens=True, # splits long texts into chunks
        return_length=True
    )
    
    # Create a Dataset object which is expected by Trainer
    dataset = Dataset.from_dict({
        "input_ids": tokenized_output["input_ids"],
        "attention_mask": tokenized_output["attention_mask"]
    })
    return dataset

# Step 5: Fine-tune the Pre-trained Model 🔧
if __name__ == "__main__":
    dataset_path = "cleaned_dataset.txt"
    
    # Ensure our cleaned dataset exists before training
    if not os.path.exists(dataset_path):
        print(f"Please run 'python 01_collect_and_clean.py' first to generate '{dataset_path}'")
        exit()
        
    print("Preparing dataset for training...")
    train_dataset = prepare_dataset(dataset_path)
    
    # Data collator automatically creates 'labels' for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    print(f"Loading {model_name} model...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Define training parameters
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500, # Decreased for small datasets compared to 10_000
        save_total_limit=2,
        logging_steps=10
    )
    
    # Start training!
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training!")
    trainer.train()
    
    # Save the final fine-tuned model and tokenizer
    output_model_dir = "./fine_tuned_model"
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Training finished. Model saved to '{output_model_dir}'")
