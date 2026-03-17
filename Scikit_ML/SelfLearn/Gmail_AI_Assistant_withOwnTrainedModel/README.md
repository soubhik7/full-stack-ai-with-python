# Gmail AI Assistant (Custom Trained Model)

Fetches unread Gmail messages, generates extractive summaries using a custom PyTorch model, and emails a formatted “summary brief” back to the authenticated user. It also exposes a small FastAPI web UI + JSON endpoints to trigger scans and view the latest summaries.

## Features

- Modular layout: API, core orchestration, ML, and external service integrations
- Extractive summarization: sentence scoring with a custom-trained model checkpoint
- FastAPI app: web UI at `/` and REST endpoints for triggering scans
- Config + secrets: YAML config with `.env`/environment variable overrides
- Notebook included: training + exploration workflow in `notebooks/`

## Directory structure

```text
Gmail_AI_Assistant_withOwnTrainedModel/
├── config/
│   ├── config.yaml
│   └── logging_config.yaml
├── logs/
│   └── app.log
├── models/
│   └── model.pth
├── notebooks/
│   └── 01_model_training_and_exploration.ipynb
├── src/
│   └── gmail_assistant/
│       ├── api/
│       ├── core/
│       ├── ml/
│       ├── services/
│       └── utils/
├── main.py
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.8+
- A Google Cloud project with the Gmail API enabled
- OAuth “Desktop app” client and a downloaded `credentials.json` (for the interactive auth flow)

## Install

```bash
pip install -r requirements.txt
```

## Configuration

Default config lives in `config/config.yaml`. Sensitive values can be provided via environment variables (including via a local `.env` file).

Supported environment variables:

- `CONFIG_PATH`: override the path to the YAML config (default: `config/config.yaml`)
- `PORT`: HTTP port for the API server (default: `8000`)
- `MODEL_PATH`: model checkpoint path (default: `models/model.pth`)
- `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, `GMAIL_REFRESH_TOKEN`: credential components (optional)
- `GMAIL_TOKEN_JSON`: full Google authorized-user JSON as a string (optional)

Credential lookup order:

1. Local `token.json` in the project root
2. `GMAIL_TOKEN_JSON` (or `gmail.token_json` in YAML)
3. `GMAIL_CLIENT_ID` + `GMAIL_CLIENT_SECRET` + `GMAIL_REFRESH_TOKEN` (or YAML equivalents)

Do not commit secrets (client secret, refresh token, token JSON, or `token.json`) to source control.

## Quickstart

1. Place `credentials.json` in the project root.
2. Run interactive auth (creates `token.json` in the project root):

```bash
python main.py auth
```

3. Start the API server:

```bash
python main.py serve
```

Open `http://localhost:8000` and click “Scan for New Emails”.

## CLI usage

Run one scan and email the summary brief (without running the web server):

```bash
python main.py summarize
```

## API endpoints

- `GET /`: web UI for the latest summaries
- `GET /trigger`: scan unread emails, generate summaries, send the brief, and return JSON
- `GET /api/summaries`: return the latest summaries as JSON

## Train the model (optional)

Training uses a synthetic dataset generator and saves a checkpoint to `models/model.pth` by default:

```bash
python main.py train
```

To explore the training pipeline and model behavior, open:

- `notebooks/01_model_training_and_exploration.ipynb`

## License

MIT
