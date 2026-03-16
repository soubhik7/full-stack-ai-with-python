import torch
import numpy as np
from preprocessor import CustomTokenizer, split_sentences, pad_sequence
from model import load_model

class Summarizer:
    def __init__(self, model_path="model.pth", max_sent_len=20):
        self.model, self.vocab = load_model(model_path)
        self.tokenizer = CustomTokenizer()
        self.max_sent_len = max_sent_len

    def summarize(self, text: str, top_n: int = 3) -> str:
        sentences = split_sentences(text)
        if not sentences:
            return ""
        
        # Preprocess sentences
        encoded_sentences = []
        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            indices = self.vocab.encode(tokens)
            padded = pad_sequence(indices, self.max_sent_len)
            encoded_sentences.append(padded)
            
        # Convert to tensor and add batch dimension (model expects list of docs)
        sentences_tensor = torch.tensor(encoded_sentences, dtype=torch.long)
        
        with torch.no_grad():
            # Get scores: model([tensor]) -> [scores]
            scores = self.model([sentences_tensor])[0]
            scores = scores.cpu().numpy()
            
        # Get indices of top N scores
        # If we have fewer sentences than top_n, take all
        n = min(top_n, len(sentences))
        top_indices = np.argsort(scores)[-n:]
        # Sort indices to maintain original order
        top_indices = sorted(top_indices)
        
        summary_sentences = [sentences[i] for i in top_indices]
        return " ".join(summary_sentences)

if __name__ == "__main__":
    summarizer = Summarizer()
    test_email = """
    Subject: Project Deadline Extension
    I hope you are having a productive week.
    Regarding our project, we have decided to extend the deadline by two weeks.
    Please confirm that you have received this message and update your schedule.
    Let me know if you have any questions.
    Best regards, Project Manager.
    """
    summary = summarizer.summarize(test_email, top_n=2)
    print(f"Original Length: {len(test_email.split())} words")
    print(f"Summary Length: {len(summary.split())} words")
    print(f"Summary: {summary}")
