import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class ExtractiveSummarizer(BaseEstimator, TransformerMixin):
    """
    Custom Extractive Summarizer acting as a Scikit-Learn transformer.
    Extracts the top N sentences based on TF-IDF scoring.
    """
    def __init__(self, top_n_sentences=3):
        self.top_n_sentences = top_n_sentences

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        summaries = []
        for text in X:
            if not isinstance(text, str) or len(text.strip()) == 0:
                summaries.append("")
                continue

            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
            
            if len(sentences) <= self.top_n_sentences:
                summaries.append(text + "\n\n(Note: This email is already shorter than the summary limit, so we couldn't condense it further!)")
                continue

            vectorizer = TfidfVectorizer(stop_words='english')
            try:
                tfidf_matrix = vectorizer.fit_transform(sentences)
                sentence_scores = []
                for i, sent in enumerate(sentences):
                    # sum of tf-idf scores for the sentence
                    score = tfidf_matrix[i].toarray().flatten().sum()
                    sentence_scores.append((score, i, sent))

                # Sort by score descending
                sentence_scores.sort(key=lambda x: x[0], reverse=True)
                # Take top N sentences and sort by original index
                top_sentences = sorted(sentence_scores[:self.top_n_sentences], key=lambda x: x[1])
                summary = " ".join([sent[2] for sent in top_sentences])
                summaries.append(summary)
            except Exception as e:
                summaries.append(text)

        return summaries

    def predict(self, X, y=None):
        # Alias for transform so it fits the model.predict(X) pattern inside app.py
        return self.transform(X)
