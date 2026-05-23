import pandas as pd
import random
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
from src.gmail_assistant.ml.preprocessor import CustomTokenizer, Vocabulary, split_sentences, pad_sequence
from src.gmail_assistant.utils.logger import get_logger

logger = get_logger(__name__)

class EmailDataset(Dataset):
    """
    A custom PyTorch Dataset for email summarization.
    It takes raw texts and their corresponding labels, then pre-processes them.
    """
    def __init__(self, texts: List[str], labels: List[List[int]], vocab: Vocabulary, tokenizer: CustomTokenizer, max_sent_len: int = 20):
        """
        Initialize the dataset.
        
        Args:
            texts: List of raw email texts.
            labels: List of labels (0 or 1) for each sentence in each email.
            vocab: The vocabulary object to use for encoding tokens.
            tokenizer: The tokenizer object to use for tokenization.
            max_sent_len: The maximum length for each sentence (in tokens).
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_sent_len = max_sent_len

    def __len__(self) -> int:
        """Returns the total number of emails in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves an email sample and its labels.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            A dictionary containing 'sentences' (torch.Tensor) and 'labels' (torch.Tensor).
        """
        text = self.texts[idx]
        sentences = split_sentences(text)
        
        # Ensure the labels match the number of sentences extracted from the text
        num_sents = len(sentences)
        labels = self.labels[idx][:num_sents]
        if len(labels) < num_sents:
            labels += [0] * (num_sents - len(labels))
        
        # Tokenize, encode, and pad each sentence
        encoded_sentences = []
        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            indices = self.vocab.encode(tokens)
            padded = pad_sequence(indices, self.max_sent_len)
            encoded_sentences.append(padded)
            
        return {
            'sentences': torch.tensor(encoded_sentences, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
    """
    A custom collate function for DataLoader.
    It returns lists of tensors for the model to handle variable sentence counts.
    """
    return {
        'sentences': [item['sentences'] for item in batch],
        'labels': [item['labels'] for item in batch]
    }

def generate_synthetic_email() -> Tuple[str, List[int]]:
    """
    Generates a single synthetic email with extractive labels.
    A label of 1 indicates an important sentence, while 0 is not.
    """
    subjects = ["Meeting Update", "Invoice #1234", "Project Status", "Lunch Plans", "Quick Question"]
    senders = ["Alice <alice@example.com>", "Bob <bob@corp.com>", "Charlie <charlie@startup.io>"]
    
    important_keywords = ["deadline", "meeting", "invoice", "payment", "please", "urgent", "required", "call", "thanks", "confirm"]
    filler_phrases = [
        "I hope you are doing well.", 
        "Just checking in on a few things.", 
        "Let me know if you need anything.", 
        "Have a great day!", 
        "Best regards,"
    ]
    
    sentences = []
    labels = []
    
    # Subject sentence is always considered important
    sentences.append(f"Subject: {random.choice(subjects)}")
    labels.append(1)
    
    # Randomly add 3-7 sentences for the body
    num_body_sents = random.randint(3, 7)
    for _ in range(num_body_sents):
        is_important = random.random() > 0.6
        if is_important:
            keyword = random.choice(important_keywords)
            sentence = f"Please {keyword} the latest update for our project."
            sentences.append(sentence)
            labels.append(1)
        else:
            sentence = random.choice(filler_phrases)
            sentences.append(sentence)
            labels.append(0)
            
    email_text = " ".join(sentences)
    return email_text, labels

def create_synthetic_dataset(num_samples: int = 1000) -> pd.DataFrame:
    """
    Creates a pandas DataFrame with a synthetic dataset of emails and labels.
    """
    logger.info(f"Generating synthetic dataset with {num_samples} samples...")
    data = []
    for _ in range(num_samples):
        text, labels = generate_synthetic_email()
        data.append({"text": text, "labels": labels})
    return pd.DataFrame(data)
