from typing import List, Dict, Any, Optional
from src.gmail_assistant.services.auth_service import AuthService
from src.gmail_assistant.services.gmail_service import GmailService
from src.gmail_assistant.ml.inference import Summarizer
from src.gmail_assistant.core.processor import EmailProcessor
from src.gmail_assistant.utils.logger import get_logger
from src.gmail_assistant.utils.config_loader import get_config

logger = get_logger(__name__)

class GmailAIAssistant:
    """
    The main application class that integrates all services and business logic.
    Provides a high-level API for triggering the email assistant's workflow.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GmailAIAssistant with provided configuration.
        """
        self.config = config or get_config()
        self.auth_service = AuthService(self.config.get('gmail'))
        
        # Initialize Summarizer: it will handle model loading internally
        model_path = self.config.get('ml.model_path')
        self.summarizer = Summarizer(model_path=model_path)
        
        logger.info("Gmail AI Assistant initialized.")

    def run_scan_and_notify(self, max_emails: int = 5) -> List[Dict[str, str]]:
        """
        Executes a complete scan of unread emails, generates summaries, and sends a brief.
        
        Args:
            max_emails: Maximum number of unread emails to process.
            
        Returns:
            A list of summaries for the processed emails.
        """
        logger.info("Running scan and notification workflow...")
        
        # 1. Obtain credentials
        creds = self.auth_service.get_credentials()
        if not creds:
            logger.error("No valid credentials found. Please run the interactive authentication.")
            return []
            
        # 2. Initialize Gmail service and processor
        gmail_service = GmailService(credentials=creds)
        processor = EmailProcessor(gmail_service, self.summarizer)
        
        # 3. Process unread emails
        summaries = processor.process_latest_unread(max_emails=max_emails)
        
        # 4. Send the summary brief email if any summaries were generated
        if summaries:
            user_email = gmail_service.get_user_email()
            html_brief = processor.generate_summary_brief_html(summaries)
            gmail_service.send_email(
                to=user_email,
                subject="Your AI Email Summary Brief",
                body_html=html_brief
            )
            logger.info(f"Summary brief sent to {user_email}.")
            
        return summaries

    def authenticate(self):
        """Triggers the interactive OAuth2 flow."""
        return self.auth_service.authenticate_interactively()
