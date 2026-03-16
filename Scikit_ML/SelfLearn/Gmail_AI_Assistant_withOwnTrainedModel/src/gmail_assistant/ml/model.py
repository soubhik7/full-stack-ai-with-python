import torch
import torch.nn as nn
import os
from typing import List, Dict, Tuple, Any
from src.gmail_assistant.utils.logger import get_logger

logger = get_logger(__name__)

class SentenceEncoder(nn.Module):
    """
    Encodes a sequence of word indices into a fixed-size sentence vector.
    Uses an Embedding layer followed by a Bidirectional LSTM and Max Pooling.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 100, hidden_dim: int = 128, n_layers: int = 1):
        """
        Initialize the sentence encoder.
        
        Args:
            vocab_size: Size of the vocabulary.
            embed_dim: Dimension of the word embeddings.
            hidden_dim: Dimension of the LSTM hidden state.
            n_layers: Number of LSTM layers.
        """
        super(SentenceEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the sentence encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) containing word indices.
            
        Returns:
            Output tensor of shape (batch_size, 2 * hidden_dim) representing sentence vectors.
        """
        # x shape: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))
        # lstm_out shape: (batch_size, seq_len, 2 * hidden_dim)
        lstm_out, _ = self.lstm(embedded)
        # Global Max Pooling over time to get sentence representation
        # out shape: (batch_size, 2 * hidden_dim)
        out, _ = torch.max(lstm_out, dim=1)
        return out

class ExtractiveSummarizer(nn.Module):
    """
    Extractive Summarizer that scores each sentence based on its content and document context.
    It combines a sentence encoder with a document-level Bi-LSTM for contextual understanding.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 100, hidden_dim: int = 128):
        """
        Initialize the summarizer.
        
        Args:
            vocab_size: Size of the vocabulary.
            embed_dim: Dimension of the word embeddings.
            hidden_dim: Dimension of the LSTM hidden state.
        """
        super(ExtractiveSummarizer, self).__init__()
        self.sentence_encoder = SentenceEncoder(vocab_size, embed_dim, hidden_dim)
        
        # Document context layer: processes sentence vectors within a document
        self.doc_lstm = nn.LSTM(2 * hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        
        # Classification head: scores each sentence
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, documents_batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass for a batch of documents.
        
        Args:
            documents_batch: List of tensors, each of shape (num_sentences, max_sent_len).
            
        Returns:
            A list of tensors, each containing sentence scores for a single document.
        """
        all_scores = []
        for doc in documents_batch:
            # doc: (num_sentences, max_sent_len)
            sent_vectors = self.sentence_encoder(doc) # (num_sentences, 2 * hidden_dim)
            
            # Add batch dimension for doc_lstm: (1, num_sentences, 2 * hidden_dim)
            doc_out, _ = self.doc_lstm(sent_vectors.unsqueeze(0))
            # doc_out: (1, num_sentences, 2 * hidden_dim)
            
            # Global document context (Max Pooling over all sentences in doc)
            doc_context, _ = torch.max(doc_out, dim=1) # (1, 2 * hidden_dim)
            # Repeat doc_context for each sentence
            doc_context_repeated = doc_context.repeat(sent_vectors.size(0), 1) # (num_sentences, 2 * hidden_dim)
            
            # Combine sentence vector and document context
            combined = torch.cat([sent_vectors, doc_context_repeated], dim=1) # (num_sentences, 4 * hidden_dim)
            
            scores = self.classifier(combined) # (num_sentences, 1)
            all_scores.append(scores.squeeze(1))
            
        return all_scores

def save_model(model: nn.Module, vocab: Any, path: str, hyperparams: Dict[str, Any]):
    """
    Saves the model state, vocabulary, and hyperparameters to a file.
    
    Args:
        model: The trained model to save.
        vocab: The vocabulary object.
        path: Path where to save the checkpoint.
        hyperparams: Dictionary of model hyperparameters.
    """
    logger.info(f"Saving model checkpoint to {path}...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_word2idx': vocab.word2idx,
        'vocab_idx2word': vocab.idx2word,
        'hyperparams': hyperparams
    }
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    logger.info("Model saved successfully.")

def load_model(path: str) -> Tuple[nn.Module, Any]:
    """
    Loads the model and vocabulary from a saved checkpoint.
    
    Args:
        path: Path to the saved checkpoint file.
        
    Returns:
        A tuple containing (model, vocabulary).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at {path}")

    logger.info(f"Loading model checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    
    from src.gmail_assistant.ml.preprocessor import Vocabulary
    vocab = Vocabulary()
    vocab.word2idx = checkpoint['vocab_word2idx']
    vocab.idx2word = checkpoint['vocab_idx2word']
    
    hp = checkpoint.get('hyperparams', {'vocab_size': len(vocab), 'embed_dim': 100, 'hidden_dim': 128})
    model = ExtractiveSummarizer(hp['vocab_size'], hp['embed_dim'], hp['hidden_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("Model and vocabulary loaded successfully.")
    return model, vocab
