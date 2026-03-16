import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import os

from preprocessor import CustomTokenizer, Vocabulary, split_sentences, pad_sequence
from model import ExtractiveSummarizer, save_model
from data_loader import create_dataset

# Hyperparameters
EMBED_DIM = 100
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 8 # Small batch because each item is a document with multiple sentences
MAX_SENT_LEN = 20
MIN_FREQ = 1

class EmailDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        sentences = split_sentences(text)
        
        # In case the labels and sentences don't match (due to synthetic gen or regex mismatch)
        # we truncate/pad to match
        num_sents = len(sentences)
        labels = self.labels[idx][:num_sents]
        if len(labels) < num_sents:
            labels += [0] * (num_sents - len(labels))
        
        # Tokenize and encode each sentence
        encoded_sentences = []
        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            indices = self.vocab.encode(tokens)
            padded = pad_sequence(indices, MAX_SENT_LEN)
            encoded_sentences.append(padded)
            
        return {
            'sentences': torch.tensor(encoded_sentences, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

def collate_fn(batch):
    # Each item in batch is a dict with 'sentences' (num_sents, max_sent_len) and 'labels' (num_sents)
    # We return lists for the model forward pass to handle variable num_sents per document
    return {
        'sentences': [item['sentences'] for item in batch],
        'labels': [item['labels'] for item in batch]
    }

def train():
    print("Generating synthetic dataset...")
    df = create_dataset(1500)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    tokenizer = CustomTokenizer()
    vocab = Vocabulary(min_freq=MIN_FREQ)
    
    print("Building vocabulary...")
    vocab.build_vocab(train_df['text'].tolist(), tokenizer)
    print(f"Vocabulary size: {len(vocab)}")
    
    train_dataset = EmailDataset(train_df['text'].tolist(), train_df['labels'].tolist(), vocab, tokenizer)
    val_dataset = EmailDataset(val_df['text'].tolist(), val_df['labels'].tolist(), vocab, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = ExtractiveSummarizer(len(vocab), EMBED_DIM, HIDDEN_DIM)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    print("Starting training loop...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass: model handles list of documents
            outputs = model(batch['sentences']) # List of scores (num_sents,)
            
            loss = 0
            for out, target in zip(outputs, batch['labels']):
                loss += criterion(out, target)
            
            loss = loss / len(batch['sentences']) # Average loss over batch
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['sentences'])
                loss = 0
                for out, target in zip(outputs, batch['labels']):
                    loss += criterion(out, target)
                val_loss += (loss / len(batch['sentences'])).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(os.path.dirname(__file__), "model.pth")
            save_model(model, vocab, model_path)
            print(f"Model saved to {model_path}! ✅")

if __name__ == "__main__":
    train()
