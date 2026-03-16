# 📧 Gmail AI Assistant: Production-Ready AI Email Summarizer

An enterprise-grade, modular, and AI-powered assistant that fetches unread emails from Gmail, generates extractive summaries using a custom-trained PyTorch model, and sends a beautifully formatted summary brief back to your inbox.

---

## 🚀 Features

-   **Modular Architecture**: Clean separation of concerns with dedicated layers for API, core business logic, machine learning, and external services.
-   **AI-Powered Summarization**: Uses a custom **Extractive Summarizer** (LSTM-based) that identifies the most important sentences in each email.
-   **Enterprise Standards**: Includes robust logging, centralized configuration management, and strict Pydantic-based API validation.
-   **Gmail API Integration**: Secure OAuth2 flow with support for token refreshing and environment variable-based credentials.
-   **Interactive Exploration**: Comprehensive Jupyter notebooks for model training, data exploration, and visualization of AI scoring.
-   **Production API**: Built with **FastAPI**, featuring a clean web interface and RESTful JSON endpoints.

---

## 📂 Directory Structure

```text
Gmail_AI_Assistant/
├── config/                  # Configuration files (YAML/JSON)
├── logs/                    # Application logs
├── models/                  # Saved model checkpoints (e.g., model.pth)
├── notebooks/               # Interactive exploration and training notebooks
├── src/
│   ├── gmail_assistant/
│   │   ├── api/             # FastAPI layer (routes, schemas, app factory)
│   │   ├── core/            # Core business logic and orchestrators
│   │   ├── ml/              # Machine Learning components (model, trainer, inference)
│   │   ├── services/        # External service integrations (Gmail, Auth)
│   │   └── utils/           # Shared utilities (logging, config)
├── tests/                   # Unit and integration tests
├── main.py                  # CLI entry point for the application
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🛠️ Setup Instructions

### 1. Prerequisites

-   Python 3.8+
-   A Google Cloud Project with the **Gmail API** enabled.
-   OAuth 2.0 Client IDs (Desktop type) from the [Google Cloud Console](https://console.cloud.google.com/).
-   Download the `credentials.json` and place it in the project root.

### 2. Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Authentication

Run the interactive authentication flow to generate your `token.json`:

```bash
python main.py auth
```

### 4. Train the AI Model (Optional)

The project comes with a synthetic data generator to train the extractive summarizer from scratch:

```bash
python main.py train
```

---

## 🏃 Usage

### Start the API Server

```bash
python main.py serve
```

Access the web interface at `http://localhost:8000`.

### API Endpoints

-   `GET /`: View the latest email summaries in a web interface.
-   `GET /trigger`: Trigger a new scan of unread emails and generate summaries.
-   `GET /api/summaries`: Retrieve the latest summaries as a JSON array.

---

## 🧪 Interactive Exploration

Open the Jupyter notebook located in `notebooks/01_model_training_and_exploration.ipynb` to explore the model architecture, visualize sentence importance scores, and experiment with different summarization parameters.

---

## 🏗️ Architectural Decisions

-   **Extractive vs. Abstractive**: Extractive summarization was chosen for its reliability and lower computational cost compared to abstractive models, making it suitable for quick email processing.
-   **LSTM-based Model**: We used a hierarchical Bi-LSTM structure to capture both word-level context within sentences and sentence-level context within the entire document.
-   **Dependency Injection**: The `GmailAIAssistant` orchestrator uses service objects (Auth, Gmail, ML) to maintain a clean and testable architecture.

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.
