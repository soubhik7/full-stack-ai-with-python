import pytest
import torch
import os
import numpy as np
from preprocessor import CustomTokenizer, Vocabulary, split_sentences, pad_sequence
from model import SentenceEncoder, ExtractiveSummarizer, save_model, load_model
from data_loader import generate_synthetic_email, create_dataset
import base64
from unittest.mock import MagicMock, patch
from inference import Summarizer
from train import EmailDataset, collate_fn, train
from app import get_email_body

def test_tokenizer():
    tokenizer = CustomTokenizer()
    tokens = tokenizer.tokenize("Hello, world! This is a test.")
    assert "hello" in tokens
    assert "world" in tokens
    assert len(tokens) == 6

def test_vocabulary():
    tokenizer = CustomTokenizer()
    texts = ["Hello world", "Hello test"]
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(texts, tokenizer)
    assert len(vocab) == 5 # <PAD>, <UNK>, hello, world, test
    encoded = vocab.encode(["hello", "world", "unknown"])
    assert encoded[0] == vocab.word2idx["hello"]
    assert encoded[2] == vocab.word2idx["<UNK>"]
    decoded = vocab.decode(encoded)
    assert decoded[0] == "hello"

def test_split_sentences():
    text = "Sentence one. Sentence two! Sentence three? Last sentence."
    sentences = split_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "Sentence one."

def test_pad_sequence():
    seq = [1, 2, 3]
    padded = pad_sequence(seq, 5, pad_val=0)
    assert len(padded) == 5
    assert padded[3:] == [0, 0]
    truncated = pad_sequence(seq, 2)
    assert len(truncated) == 2
    assert truncated == [1, 2]

def test_sentence_encoder():
    vocab_size = 10
    encoder = SentenceEncoder(vocab_size, embed_dim=16, hidden_dim=8)
    x = torch.randint(0, vocab_size, (2, 5)) # (batch_size, seq_len)
    out = encoder(x)
    assert out.shape == (2, 16) # (batch_size, 2 * hidden_dim)

def test_extractive_summarizer():
    vocab_size = 10
    model = ExtractiveSummarizer(vocab_size, embed_dim=16, hidden_dim=8)
    # List of 2 documents, each with 3 sentences, each sentence of length 5
    doc1 = torch.randint(0, vocab_size, (3, 5))
    doc2 = torch.randint(0, vocab_size, (2, 5))
    batch = [doc1, doc2]
    scores = model(batch)
    assert len(scores) == 2
    assert scores[0].shape == (3,)
    assert scores[1].shape == (2,)
    assert torch.all(scores[0] >= 0) and torch.all(scores[0] <= 1)

def test_save_load_model(tmp_path):
    model_path = os.path.join(tmp_path, "test_model.pth")
    vocab = Vocabulary()
    vocab.word2idx = {"<PAD>": 0, "<UNK>": 1, "hello": 2}
    vocab.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "hello"}
    # Use default hyperparams to match load_model's default
    hp = {'vocab_size': 3, 'embed_dim': 100, 'hidden_dim': 128}
    model = ExtractiveSummarizer(hp['vocab_size'], hp['embed_dim'], hp['hidden_dim'])
    save_model(model, vocab, model_path)
    
    loaded_model, loaded_vocab = load_model(model_path)
    assert len(loaded_vocab) == 3
    assert loaded_vocab.word2idx["hello"] == 2
    assert isinstance(loaded_model, ExtractiveSummarizer)

def test_data_loader():
    text, labels = generate_synthetic_email()
    assert isinstance(text, str)
    assert isinstance(labels, list)
    assert all(l in [0, 1] for l in labels)
    
    df = create_dataset(10)
    assert len(df) == 10
    assert "text" in df.columns
    assert "labels" in df.columns

def test_email_dataset():
    tokenizer = CustomTokenizer()
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(["hello world. goodbye world."], tokenizer)
    dataset = EmailDataset(["hello world. goodbye world."], [[1, 0]], vocab, tokenizer)
    assert len(dataset) == 1
    item = dataset[0]
    assert "sentences" in item
    assert "labels" in item
    assert item["sentences"].shape == (2, 20) # 2 sentences, padded to 20

def test_get_email_body():
    # Test simple text/plain
    payload = {
        'mimeType': 'text/plain',
        'body': {'data': base64.urlsafe_b64encode(b"Hello").decode()}
    }
    assert get_email_body(payload) == "Hello"
    
    # Test multipart
    payload_multi = {
        'parts': [
            {'mimeType': 'text/html', 'body': {'data': 'html'}},
            {'mimeType': 'text/plain', 'body': {'data': base64.urlsafe_b64encode(b"Part2").decode()}}
        ]
    }
    assert get_email_body(payload_multi) == "Part2"

def test_summarizer(tmp_path):
    # Train a tiny model for testing
    model_path = os.path.join(tmp_path, "tiny_model.pth")
    vocab = Vocabulary()
    vocab.word2idx = {"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3}
    vocab.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "hello", 3: "world"}
    # Use the same hyperparams as the model's default for consistency in this test
    model = ExtractiveSummarizer(len(vocab), embed_dim=100, hidden_dim=128)
    save_model(model, vocab, model_path)
    
    summarizer = Summarizer(model_path=model_path)
    text = "Hello world. This is a test sentence."
    summary = summarizer.summarize(text, top_n=1)
    assert isinstance(summary, str)
    assert len(split_sentences(summary)) == 1

def test_save_load_model_hp(tmp_path):
    model_path = os.path.join(tmp_path, "test_hp_model.pth")
    vocab = Vocabulary()
    vocab.word2idx = {"<PAD>": 0, "<UNK>": 1, "hello": 2}
    vocab.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "hello"}
    # Test with custom hyperparams
    hp = {'vocab_size': 3, 'embed_dim': 32, 'hidden_dim': 16}
    model = ExtractiveSummarizer(hp['vocab_size'], hp['embed_dim'], hp['hidden_dim'])
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_word2idx': vocab.word2idx,
        'vocab_idx2word': vocab.idx2word,
        'hyperparams': hp
    }
    torch.save(checkpoint, model_path)
    
    loaded_model, _ = load_model(model_path)
    assert loaded_model.sentence_encoder.embedding.embedding_dim == 32

def test_app_main_full_flow():
    # Mock all external dependencies for a full flow
    with patch('app.Summarizer') as mock_summarizer:
        mock_summarizer.return_value.summarize.return_value = "Summary"
        
        with patch('os.path.exists', return_value=True):
            with patch('google.oauth2.credentials.Credentials.from_authorized_user_file') as mock_creds:
                mock_creds.return_value.valid = True
                
                with patch('app.build') as mock_build:
                    mock_service = MagicMock()
                    mock_build.return_value = mock_service
                    
                    # Mock messages.list
                    mock_service.users().messages().list().execute.return_value = {
                        'messages': [{'id': '123'}]
                    }
                    
                    # Mock messages.get
                    mock_service.users().messages().get().execute.return_value = {
                        'id': '123',
                        'snippet': 'Test snippet',
                        'payload': {
                            'headers': [{'name': 'Subject', 'value': 'Sub'}, {'name': 'From', 'value': 'Sender'}],
                            'mimeType': 'text/plain',
                            'body': {'data': base64.urlsafe_b64encode(b"Body").decode()}
                        }
                    }
                    
                    # Mock other methods
                    mock_service.users().getProfile().execute.return_value = {'emailAddress': 'test@example.com'}
                    
                    from app import main as app_main
                    app_main()
                    
                    assert mock_service.users().messages().modify.called
                    assert mock_service.users().messages().send.called

def test_auth_authenticate_refresh():
    # Test auth.authenticate with expired but refreshable token
    with patch('os.path.exists', return_value=True):
        with patch('google.oauth2.credentials.Credentials.from_authorized_user_file') as mock_creds:
            creds_instance = mock_creds.return_value
            creds_instance.valid = False
            creds_instance.expired = True
            creds_instance.refresh_token = "some_token"
            creds_instance.to_json.return_value = '{"token": "test"}'
            
            with patch('google.auth.transport.requests.Request') as mock_request:
                # Mock open to avoid writing to token.json
                with patch('builtins.open', MagicMock()):
                    from auth import authenticate
                    authenticate()
                    assert creds_instance.refresh.called

def test_train_function():
    # Mock create_dataset to return a tiny dataset
    tiny_df = create_dataset(5)
    
    # Patch everything needed to make train() run quickly and without disk I/O
    with patch('train.create_dataset', return_value=tiny_df):
        with patch('train.train_test_split', return_value=(tiny_df, tiny_df)):
            with patch('train.save_model'):
                with patch('train.EPOCHS', 1):
                    # Call train() directly
                    train()

def test_app_main_no_creds():
    # Test app.main when no creds are found
    # We need to mock os.path.exists to return False for token.json
    with patch('os.path.exists', return_value=False):
        # And os.environ to not have GMAIL_TOKEN_JSON
        with patch.dict(os.environ, {}, clear=True):
             with patch('app.Summarizer'):
                 from app import main as app_main
                 app_main()

def test_auth_authenticate():
    # Test auth.authenticate with existing token
    with patch('os.path.exists', return_value=True):
        with patch('google.oauth2.credentials.Credentials.from_authorized_user_file') as mock_creds:
            mock_creds.return_value.valid = True
            from auth import authenticate
            creds = authenticate()
            assert creds is not None
