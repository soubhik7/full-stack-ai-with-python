import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import os

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.gmail_assistant.ml.model import load_model
from src.gmail_assistant.ml.data import create_synthetic_dataset, EmailDataset, collate_fn
from src.gmail_assistant.ml.preprocessor import CustomTokenizer
from src.gmail_assistant.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model_path: str = 'models/model.pth', num_samples: int = 200):
    """
    Loads a trained model and evaluates it on a synthetic test set.
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}. Please train the model first.")
        return

    logger.info(f"Loading model from {model_path} for evaluation...")
    model, vocab = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    logger.info(f"Generating synthetic test dataset with {num_samples} samples...")
    test_df = create_synthetic_dataset(num_samples)
    tokenizer = CustomTokenizer()
    
    test_dataset = EmailDataset(
        test_df['text'].tolist(), 
        test_df['labels'].tolist(), 
        vocab, 
        tokenizer,
        max_sent_len=20
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    all_preds = []
    all_labels = []

    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            sentences = [s.to(device) for s in batch['sentences']]
            labels = [l.numpy() for l in batch['labels']]
            
            outputs = model(sentences)
            
            for out, target in zip(outputs, labels):
                preds = (out >= 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(target)

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("\n" + "="*30)
    print("      Model Evaluation Results")
    print("="*30)
    print(f"Accuracy:  {acc:.4%}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    evaluate_model()
