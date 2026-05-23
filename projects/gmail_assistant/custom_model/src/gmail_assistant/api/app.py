from fastapi import FastAPI
from src.gmail_assistant.api.routes import router, init_assistant
from src.gmail_assistant.core.assistant import GmailAIAssistant
from src.gmail_assistant.utils.logger import setup_logging, get_logger
from src.gmail_assistant.utils.config_loader import get_config

logger = get_logger(__name__)

def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.
    """
    # 1. Setup logging
    setup_logging()
    
    # 2. Load configuration
    config = get_config()
    
    # 3. Initialize the core assistant
    assistant = GmailAIAssistant()
    init_assistant(assistant)
    
    # 4. Create the FastAPI app instance
    app = FastAPI(
        title="Gmail AI Assistant",
        description="An AI-powered assistant that summarizes unread emails using a custom extractive summarization model.",
        version="1.0.0"
    )
    
    # 5. Include API routes
    app.include_router(router)
    
    logger.info("FastAPI application created successfully.")
    return app

# The app instance for uvicorn
app = create_app()
