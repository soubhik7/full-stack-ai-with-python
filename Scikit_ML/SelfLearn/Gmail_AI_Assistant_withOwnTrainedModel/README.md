# Custom AI Gmail Assistant (Trained From Scratch)

This project implements a complete custom AI model pipeline for summarizing Gmail messages. Every component—from data preprocessing and feature extraction to the neural network architecture and training loop—is authored from scratch, avoiding any pre-trained or transfer-learned models.

## Project Structure

- `app.py`: The main Gmail assistant application that integrates the custom model.
- `auth.py`: Handles Google OAuth2 authentication.
- `model.py`: Defines the custom Neural Extractive Summarizer architecture (Bi-LSTM + Global Context).
- `train.py`: Training pipeline with hyperparameters, validation, and serialization.
- `preprocessor.py`: Custom tokenizer, vocabulary, and text processing tools.
- `inference.py`: Inference API for the trained model.
- `data_loader.py`: Synthetic dataset generator for training the summarizer.
- `requirements.txt`: Project dependencies.
- `tests/`: Unit tests with ≥90% coverage.

## Model Architecture

The summarizer uses a **Neural Extractive** approach:
1.  **Sentence Encoder**: Custom Embedding layer followed by a Bi-Directional LSTM. Global Max Pooling produces a fixed-size vector for each sentence.
2.  **Document Encoder**: Sentence vectors are processed by another Bi-LSTM to capture the document's flow.
3.  **Global Context**: A document-level context vector is computed by pooling sentence representations.
4.  **Scorer**: For each sentence, its vector is concatenated with the document context and passed through a Multi-Layer Perceptron (MLP) with a Sigmoid activation to predict an "importance score".
5.  **Selection**: The top-N sentences with the highest scores are selected as the summary.

- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy (BCE)

## Reproduction Steps

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Train the Model
The training script generates a synthetic dataset and trains the model from scratch.
```bash
python3 train.py
```
This will produce `model.pth` and `vocab.pkl` (serialized within the checkpoint).

### 3. Run Unit Tests
To verify the implementation and check coverage:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 -m pytest --cov=. tests/
```

## Deployment

### 1. Gmail API Setup
1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  Create a project and enable the **Gmail API**.
3.  Configure the OAuth Consent Screen and create **OAuth 2.0 Client IDs**.
4.  Download the JSON file and save it as `credentials.json` in the project directory (`Scikit_ML/SelfLearn/Gmail_AI_Assistant_withOwnTrainedModel/`).

### 2. Authentication
Run the authentication script within the project directory to generate `token.json`:
```bash
cd Scikit_ML/SelfLearn/Gmail_AI_Assistant_withOwnTrainedModel
python3 auth.py
```

### 3. Run the Assistant
```bash
python3 app.py
```
The assistant will:
- Fetch unread emails.
- Summarize them using the custom model.
- Mark them as read.
- Send a compiled summary email back to you.

### 4. GitHub Actions Deployment
- Ensure the workflow file `.github/workflows/gmail-ai-custom.yml` exists at the root of the repository.
- Add your `GMAIL_TOKEN_JSON` (the contents of `token.json`) as a secret in your GitHub repository.
- The action will automatically run every 3 hours and process your inbox.

## Features
- **Zero Pre-trained Models**: No Transformers, no pre-trained embeddings (Word2Vec/GloVe).
- **In-house Preprocessing**: Custom regex tokenizer and vocabulary builder.
- **Custom Training Loop**: Manual implementation of the training/validation cycle.
- **High Test Coverage**: Thoroughly tested components ensuring reliability.
