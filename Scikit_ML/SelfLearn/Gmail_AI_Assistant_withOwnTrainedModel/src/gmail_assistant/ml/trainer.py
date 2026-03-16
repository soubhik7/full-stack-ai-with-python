import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from typing import Dict, Any, List, Optional
from src.gmail_assistant.ml.preprocessor import CustomTokenizer, Vocabulary
from src.gmail_assistant.ml.model import ExtractiveSummarizer, save_model
from src.gmail_assistant.ml.data import create_synthetic_dataset, EmailDataset, collate_fn
from src.gmail_assistant.utils.logger import get_logger
from src.gmail_assistant.utils.config_loader import get_config

logger = get_logger(__name__)

class ModelTrainer:
    """
    Handles the training and validation loop for the extractive summarizer model.
    It manages the training data, loss calculation, optimization, and model saving.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model trainer with provided configuration.
        
        Args:
            config: Configuration dictionary for hyperparameters. If None, it uses defaults.
        """
        self.config = config or get_config().get('ml', {})
        self.embed_dim = self.config.get('embed_dim', 100)
        self.hidden_dim = self.config.get('hidden_dim', 128)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.epochs = self.config.get('epochs', 20)
        self.batch_size = self.config.get('batch_size', 8)
        self.max_sent_len = self.config.get('max_sent_len', 20)
        self.min_freq = self.config.get('min_freq', 1)
        self.model_path = self.config.get('model_path', 'models/model.pth')
        
        self.tokenizer = CustomTokenizer()
        self.vocab = Vocabulary(min_freq=self.min_freq)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Trainer initialized using device: {self.device}.")

    def prepare_data(self, num_samples: int = 1500) -> Tuple[DataLoader, DataLoader]:
        """
        Generates synthetic data, splits it into training/validation sets, and creates DataLoaders.
        """
        df = create_synthetic_dataset(num_samples)
        
        # Split into training and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Build the vocabulary from the training data
        logger.info("Building vocabulary...")
        self.vocab.build_vocab(train_df['text'].tolist(), self.tokenizer)
        logger.info(f"Vocabulary built with {len(self.vocab)} tokens.")
        
        # Create Dataset objects
        train_dataset = EmailDataset(
            train_df['text'].tolist(), 
            train_df['labels'].tolist(), 
            self.vocab, 
            self.tokenizer,
            self.max_sent_len
        )
        val_dataset = EmailDataset(
            val_df['text'].tolist(), 
            val_df['labels'].tolist(), 
            self.vocab, 
            self.tokenizer,
            self.max_sent_len
        )
        
        # Create DataLoader objects
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader

    def train(self, num_samples: int = 1500):
        """
        The main training loop.
        """
        train_loader, val_loader = self.prepare_data(num_samples)
        
        # Initialize the model, criterion, and optimizer
        model = ExtractiveSummarizer(len(self.vocab), self.embed_dim, self.hidden_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        best_val_loss = float('inf')
        
        logger.info("Starting training loop...")
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move sentence tensors to device
                sentences = [s.to(self.device) for s in batch['sentences']]
                labels = [l.to(self.device) for l in batch['labels']]
                
                # Forward pass: model handles a list of documents
                outputs = model(sentences) # List of scores (num_sents,)
                
                # Calculate loss for the batch
                loss = 0
                for out, target in zip(outputs, labels):
                    loss += criterion(out, target)
                
                loss = loss / len(sentences) # Average loss over documents in batch
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    sentences = [s.to(self.device) for s in batch['sentences']]
                    labels = [l.to(self.device) for l in batch['labels']]
                    
                    outputs = model(sentences)
                    loss = 0
                    for out, target in zip(outputs, labels):
                        loss += criterion(out, target)
                    val_loss += (loss / len(sentences)).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            # Save the model if validation loss improves
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                hyperparams = {
                    'vocab_size': len(self.vocab),
                    'embed_dim': self.embed_dim,
                    'hidden_dim': self.hidden_dim
                }
                save_model(model, self.vocab, self.model_path, hyperparams)
                logger.info(f"Model checkpoint saved with Val Loss: {avg_val_loss:.4f}! ✅")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train(num_samples=1500)
