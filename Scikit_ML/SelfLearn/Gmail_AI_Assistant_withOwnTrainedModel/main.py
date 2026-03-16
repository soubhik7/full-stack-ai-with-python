import uvicorn
import os
import argparse
from src.gmail_assistant.utils.config_loader import get_config
from src.gmail_assistant.utils.logger import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

def run_server():
    """Starts the FastAPI web server."""
    config = get_config()
    port = int(os.environ.get("PORT", config.get('api.port', 8000)))
    host = config.get('api.host', '0.0.0.0')
    
    logger.info(f"Starting Gmail AI Assistant server on {host}:{port}...")
    uvicorn.run("src.gmail_assistant.api.app:app", host=host, port=port, reload=True)

def authenticate():
    """Runs the interactive OAuth2 authentication flow."""
    from src.gmail_assistant.core.assistant import GmailAIAssistant
    logger.info("Starting interactive authentication flow...")
    assistant = GmailAIAssistant()
    assistant.authenticate()
    logger.info("Authentication complete. You can now start the server.")

def train_model():
    """Trains the extractive summarization model from scratch."""
    from src.gmail_assistant.ml.trainer import ModelTrainer
    logger.info("Starting model training process...")
    trainer = ModelTrainer()
    trainer.train(num_samples=1500)
    logger.info("Model training completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gmail AI Assistant CLI")
    parser.add_argument("command", choices=["serve", "auth", "train"], help="Command to run")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        run_server()
    elif args.command == "auth":
        authenticate()
    elif args.command == "train":
        train_model()
