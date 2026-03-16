import os
import json
from typing import Optional, List, Dict, Any
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from src.gmail_assistant.utils.logger import get_logger
from src.gmail_assistant.utils.config_loader import get_config

logger = get_logger(__name__)

class AuthService:
    """
    Handles authentication and credential management for the Gmail API.
    Supports local token storage and environment variable-based credentials.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AuthService with provided configuration.
        """
        self.config = config or get_config().get('gmail', {})
        self.scopes = self.config.get('scopes', ['https://www.googleapis.com/auth/gmail.modify'])
        self.token_path = 'token.json'
        self.credentials_path = 'credentials.json'

    def get_credentials(self) -> Optional[Credentials]:
        """
        Retrieves valid Gmail API credentials from available sources.
        """
        creds = None
        
        # 1. Try loading from local token.json
        if os.path.exists(self.token_path):
            try:
                creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)
                logger.info("Loaded credentials from token.json.")
            except Exception as e:
                logger.error(f"Error loading token.json: {e}")
        
        # 2. Try loading from environment variable (GMAIL_TOKEN_JSON)
        if not creds:
            token_json = self.config.get('token_json', "").strip()
            if token_json:
                try:
                    token_data = json.loads(token_json)
                    creds = Credentials.from_authorized_user_info(token_data, self.scopes)
                    logger.info("Loaded credentials from GMAIL_TOKEN_JSON environment variable.")
                except Exception as e:
                    logger.error(f"Failed to decode GMAIL_TOKEN_JSON: {e}")
        
        # 3. Try building from individual secrets (ID, Secret, Refresh Token)
        if not creds:
            client_id = self.config.get('client_id')
            client_secret = self.config.get('client_secret')
            refresh_token = self.config.get('refresh_token')
            
            if client_id and client_secret and refresh_token:
                logger.info("Building credentials from individual GMAIL secrets.")
                token_data = {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
                creds = Credentials.from_authorized_user_info(token_data, self.scopes)
            else:
                logger.warning("No credentials found in environment or local files.")

        # 4. Refresh credentials if expired
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("Credentials refreshed successfully.")
            except Exception as e:
                logger.error(f"Error refreshing credentials: {e}")
                creds = None

        return creds

    def authenticate_interactively(self) -> Credentials:
        """
        Performs interactive OAuth2 flow to obtain credentials.
        Requires credentials.json to be present in the root directory.
        """
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"credentials.json not found at {self.credentials_path}. Please download it from Google Cloud Console.")

        flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.scopes)
        creds = flow.run_local_server(port=0)
        
        # Save the credentials for future use
        with open(self.token_path, 'w') as token_file:
            token_file.write(creds.to_json())
        
        logger.info(f"Authentication successful. Token saved to {self.token_path}.")
        return creds
