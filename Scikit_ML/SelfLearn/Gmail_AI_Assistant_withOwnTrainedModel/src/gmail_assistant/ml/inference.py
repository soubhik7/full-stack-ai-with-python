import torch
import numpy as np
import os
from typing import List, Optional
from src.gmail_assistant.ml.preprocessor import CustomTokenizer, split_sentences, pad_sequence
from src.gmail_assistant.ml.model import load_model
from src.gmail_assistant.utils.logger import get_logger

logger = get_logger(__name__)

class Summarizer:
    """
    A high-level interface for performing extractive summarization on raw text.
    It handles preprocessing, model inference, and summary generation.
    """
    def __init__(self, model_path: Optional[str] = None, max_sent_len: int = 20):
        """
        Initialize the summarizer.
        
        Args:
            model_path: Path to the saved model checkpoint. If None, it uses the default path.
            max_sent_len: Maximum length of a sentence (in tokens).
        """
        if model_path is None:
            # Try loading from the models/ directory if not provided
            model_path = "models/model.pth"
            if not os.path.exists(model_path):
                # Fallback to the current directory if needed
                model_path = "model.pth"
        
        try:
            self.model, self.vocab = load_model(model_path)
            logger.info(f"Summarizer initialized using model at {model_path}.")
        except Exception as e:
            logger.error(f"Failed to initialize Summarizer: {e}")
            raise e
            
        self.tokenizer = CustomTokenizer()
        self.max_sent_len = max_sent_len

    def summarize(self, text: str, top_n: int = 3) -> str:
        """
        Generates an extractive summary for the input text.
        
        Args:
            text: The raw text to summarize.
            top_n: The number of top-scored sentences to include in the summary.
            
        Returns:
            The generated summary as a single string.
        """
        if not text:
            return ""
            
        sentences = split_sentences(text)
        if not sentences:
            return ""
        
        # Preprocess each sentence: tokenize, encode, and pad
        encoded_sentences = []
        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            indices = self.vocab.encode(tokens)
            padded = pad_sequence(indices, self.max_sent_len)
            encoded_sentences.append(padded)
            
        # Convert the processed sentences to a PyTorch tensor
        # Shape: (num_sentences, max_sent_len)
        sentences_tensor = torch.tensor(encoded_sentences, dtype=torch.long)
        
        with torch.no_grad():
            # Get sentence scores from the model.
            # The model expects a list of documents (each a tensor of sentences).
            scores = self.model([sentences_tensor])[0]
            scores = scores.cpu().numpy()
            
        # Select the indices of the top N sentences based on their scores.
        n = min(top_n, len(sentences))
        top_indices = np.argsort(scores)[-n:]
        
        # Re-sort indices to maintain the original order of sentences.
        top_indices = sorted(top_indices)
        
        # Combine the selected sentences into the final summary.
        summary_sentences = [sentences[i] for i in top_indices]
        summary = " ".join(summary_sentences)
        
        return summary
