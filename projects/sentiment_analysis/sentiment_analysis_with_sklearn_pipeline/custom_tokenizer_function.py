"""
custom_tokenizer_function.py
=============================
This module contains the CustomTokenizer class used for text cleaning
in our Sentiment Analysis pipeline.

What does it do?
- Removes stopwords (common words like "the", "is", "and" that don't carry meaning)
- Removes punctuation (!, ?, . etc.)
- Applies lemmatization (converts words to their base form, e.g. "running" → "run")

Why do we need this?
- Machine Learning models work better with clean, meaningful text
- Stopwords and punctuation add noise and don't help in sentiment classification
- Lemmatization reduces vocabulary size and groups similar words together
"""

import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS

# ──────────────────────────────────────────────────────────────────────
# Load the spaCy English model (small version)
# This model knows English grammar, word forms, and can tokenize text.
# If you don't have it installed, run:  python -m spacy download en_core_web_sm
# ──────────────────────────────────────────────────────────────────────
nlp = spacy.load('en_core_web_sm')

# Get all punctuation characters as a string, e.g. !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
punct = string.punctuation

# Convert spaCy's set of stopwords to a list for easy membership checking
stopwords = list(STOP_WORDS)


class CustomTokenizer():
    """
    A custom tokenizer class that cleans text data for our ML pipeline.
    
    This class is designed to be used with sklearn's TfidfVectorizer.
    The TfidfVectorizer accepts a custom tokenizer function, and we provide
    our text_data_cleaning method for this purpose.
    
    Example usage:
        tokenizer = CustomTokenizer()
        clean_tokens = tokenizer.text_data_cleaning("I love my Alexa!")
        # Output: ['love', 'alexa']  (stopwords like "I" and "my" are removed)
    """
    
    def __init__(self):
        """Initialize the tokenizer. No special setup needed."""
        pass

    def text_data_cleaning(self, sentence):
        """
        Clean and tokenize a single sentence/review.
        
        Steps:
        1. Use spaCy to tokenize the sentence into individual words
        2. Convert each word to its base form (lemma) in lowercase
        3. Remove stopwords and punctuation
        
        Args:
            sentence (str): A raw review text, e.g. "I absolutely love my Echo!"
            
        Returns:
            list: A list of cleaned tokens, e.g. ['absolutely', 'love', 'echo']
        """
        # Step 1: spaCy processes the sentence — tokenizes it and analyzes each word
        doc = nlp(sentence)

        # Step 2: Convert each word to its base form (lemmatization)
        tokens = []
        for token in doc:
            if token.lemma_ != "-PRON-":
                # If the word is NOT a pronoun, use its lemma (base form) in lowercase
                # Example: "running" → "run", "better" → "good"
                temp = token.lemma_.lower().strip()
            else:
                # If the word IS a pronoun (he, she, it, etc.), just use lowercase
                # because pronouns don't have a meaningful lemma
                temp = token.lower_
            tokens.append(temp)

        # Step 3: Remove stopwords and punctuation
        cleaned_tokens = []
        for token in tokens:
            # Only keep the token if it's NOT a stopword AND NOT punctuation
            if token not in stopwords and token not in punct:
                cleaned_tokens.append(token)
        
        return cleaned_tokens