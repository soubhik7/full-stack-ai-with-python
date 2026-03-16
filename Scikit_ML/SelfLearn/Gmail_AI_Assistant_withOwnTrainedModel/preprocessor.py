import re
import numpy as np
import torch
import collections
from typing import List, Dict, Tuple

class CustomTokenizer:
    """A simple custom tokenizer built from scratch."""
    def __init__(self, lower=True):
        self.lower = lower

    def tokenize(self, text: str) -> List[str]:
        if self.lower:
            text = text.lower()
        # Basic regex-based tokenization
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

class Vocabulary:
    """A custom vocabulary to map tokens to indices."""
    def __init__(self, min_freq=2, max_size=10000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.freqs = collections.Counter()

    def build_vocab(self, texts: List[str], tokenizer: CustomTokenizer):
        for text in texts:
            tokens = tokenizer.tokenize(text)
            self.freqs.update(tokens)
        
        # Filter by frequency and size
        sorted_words = sorted(
            [w for w, f in self.freqs.items() if f >= self.min_freq],
            key=lambda x: self.freqs[x],
            reverse=True
        )[:self.max_size]

        for word in sorted_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2word.get(idx, "<UNK>") for idx in indices]

    def __len__(self):
        return len(self.word2idx)

def split_sentences(text: str) -> List[str]:
    """Split text into sentences manually using regex."""
    # Simple regex for sentence splitting: matches . ! ? followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def pad_sequence(seq: List[int], max_len: int, pad_val: int = 0) -> List[int]:
    """Pad or truncate a sequence to max_len."""
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [pad_val] * (max_len - len(seq))
