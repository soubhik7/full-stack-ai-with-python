import re
import collections
from typing import List, Dict, Tuple
import torch
from src.gmail_assistant.utils.logger import get_logger

logger = get_logger(__name__)

class CustomTokenizer:
    """
    A simple custom tokenizer for text processing.
    It performs basic tokenization by finding all word-like tokens.
    """
    def __init__(self, lower: bool = True):
        """
        Initialize the tokenizer.
        
        Args:
            lower: Whether to lowercase the text before tokenization.
        """
        self.lower = lower

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        
        Args:
            text: The raw text to tokenize.
            
        Returns:
            A list of tokens (strings).
        """
        if not text:
            return []
        
        if self.lower:
            text = text.lower()
        
        # Basic regex-based tokenization: matches word characters
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

class Vocabulary:
    """
    Manages a mapping between tokens and unique integer indices.
    Includes special tokens for padding and unknown words.
    """
    def __init__(self, min_freq: int = 2, max_size: int = 10000):
        """
        Initialize the vocabulary.
        
        Args:
            min_freq: Minimum frequency for a word to be included in the vocabulary.
            max_size: Maximum number of words in the vocabulary.
        """
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.freqs = collections.Counter()

    def build_vocab(self, texts: List[str], tokenizer: CustomTokenizer):
        """
        Build the vocabulary from a list of texts.
        
        Args:
            texts: List of strings to build the vocabulary from.
            tokenizer: The tokenizer to use for splitting text into tokens.
        """
        logger.info("Building vocabulary from provided texts...")
        for text in texts:
            tokens = tokenizer.tokenize(text)
            self.freqs.update(tokens)
        
        # Filter words by frequency and then by maximum size
        sorted_words = sorted(
            [w for w, f in self.freqs.items() if f >= self.min_freq],
            key=lambda x: self.freqs[x],
            reverse=True
        )[:self.max_size]

        for word in sorted_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        logger.info(f"Vocabulary built with {len(self.word2idx)} unique tokens.")

    def encode(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of tokens to their corresponding indices.
        
        Args:
            tokens: List of tokens to encode.
            
        Returns:
            List of integer indices.
        """
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """
        Convert a list of indices back to their corresponding tokens.
        
        Args:
            indices: List of indices to decode.
            
        Returns:
            List of tokens (strings).
        """
        return [self.idx2word.get(idx, "<UNK>") for idx in indices]

    def __len__(self) -> int:
        """Returns the number of unique tokens in the vocabulary."""
        return len(self.word2idx)

def split_sentences(text: str) -> List[str]:
    """
    Split text into individual sentences using regular expressions.
    
    Args:
        text: The raw text to split.
        
    Returns:
        A list of sentence strings.
    """
    if not text:
        return []
    
    # Simple regex for sentence splitting: matches . ! ? followed by space or end of string
    # Uses positive lookbehind to keep the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def pad_sequence(seq: List[int], max_len: int, pad_val: int = 0) -> List[int]:
    """
    Pad or truncate a sequence of indices to a fixed length.
    
    Args:
        seq: The list of indices to pad or truncate.
        max_len: The target length.
        pad_val: The value to use for padding.
        
    Returns:
        A list of indices of length max_len.
    """
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [pad_val] * (max_len - len(seq))
