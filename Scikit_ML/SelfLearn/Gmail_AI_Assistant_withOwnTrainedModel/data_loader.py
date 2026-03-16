import pandas as pd
import random
from typing import List, Tuple

def generate_synthetic_email() -> Tuple[str, List[int]]:
    """
    Generates a synthetic email and labels for extractive summarization.
    1 means sentence is important, 0 means not.
    """
    subjects = ["Meeting Update", "Invoice #1234", "Project Status", "Lunch Plans", "Quick Question"]
    senders = ["Alice <alice@example.com>", "Bob <bob@corp.com>", "Charlie <charlie@startup.io>"]
    
    important_keywords = ["deadline", "meeting", "invoice", "payment", "please", "urgent", "required", "call", "thanks", "confirm"]
    filler_phrases = ["I hope you are doing well.", "Just checking in on a few things.", "Let me know if you need anything.", "Have a great day!", "Best regards,"]
    
    sentences = []
    labels = []
    
    # Randomly pick a subject and sender-like intro
    sentences.append(f"Subject: {random.choice(subjects)}")
    labels.append(1)
    
    # Randomly add 3-7 sentences
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

def create_dataset(num_samples=1000) -> pd.DataFrame:
    data = []
    for _ in range(num_samples):
        text, labels = generate_synthetic_email()
        data.append({"text": text, "labels": labels})
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_dataset(100)
    print(df.head())
    print(f"Dataset size: {len(df)}")
