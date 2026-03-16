import torch
import torch.nn as nn
import torch.nn.functional as F

class SentenceEncoder(nn.Module):
    """Encodes a sequence of word indices into a fixed-size sentence vector."""
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, n_layers=1):
        super(SentenceEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
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
    """
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128):
        super(ExtractiveSummarizer, self).__init__()
        self.sentence_encoder = SentenceEncoder(vocab_size, embed_dim, hidden_dim)
        
        # Document context layer
        # Sentence vectors (2*hidden_dim) -> Bi-LSTM -> (2*hidden_dim)
        self.doc_lstm = nn.LSTM(2 * hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        
        # Classification head
        # Concatenate sentence vector (2*hidden_dim) and doc context (2*hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, sentences_batch):
        """
        sentences_batch: List of tensors, each tensor (num_sentences, max_sent_len)
        Returns: Scores for each sentence in each document.
        """
        # We handle one document at a time for simplicity in this "from scratch" version
        # but let's try to be batch-aware if possible.
        # Actually, for variable number of sentences, it's easier to process one document.
        
        all_scores = []
        for doc in sentences_batch:
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

def save_model(model, vocab, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_word2idx': vocab.word2idx,
        'vocab_idx2word': vocab.idx2word,
        'hyperparams': {
            'vocab_size': len(vocab),
            'embed_dim': 100,
            'hidden_dim': 128
        }
    }
    torch.save(checkpoint, path)

def load_model(path):
    checkpoint = torch.load(path)
    from preprocessor import Vocabulary
    vocab = Vocabulary()
    vocab.word2idx = checkpoint['vocab_word2idx']
    vocab.idx2word = checkpoint['vocab_idx2word']
    
    hp = checkpoint.get('hyperparams', {'vocab_size': len(vocab), 'embed_dim': 100, 'hidden_dim': 128})
    model = ExtractiveSummarizer(hp['vocab_size'], hp['embed_dim'], hp['hidden_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, vocab
