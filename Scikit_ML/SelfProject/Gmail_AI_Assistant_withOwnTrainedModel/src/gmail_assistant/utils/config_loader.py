import os
import yaml
import logging
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ConfigLoader:
    """
    A utility class to load configuration from YAML files and environment variables.
    Follows the Singleton pattern to ensure configuration is loaded only once.
    """
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Loads configuration from config/config.yaml and overrides with environment variables."""
        config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            logging.warning(f"Configuration file not found at {config_path}. Using environment variables only.")

        # Override with environment variables for sensitive data or dynamic settings
        self._config['gmail'] = self._config.get('gmail', {})
        self._config['gmail']['client_id'] = os.getenv("GMAIL_CLIENT_ID", self._config['gmail'].get('client_id'))
        self._config['gmail']['client_secret'] = os.getenv("GMAIL_CLIENT_SECRET", self._config['gmail'].get('client_secret'))
        self._config['gmail']['refresh_token'] = os.getenv("GMAIL_REFRESH_TOKEN", self._config['gmail'].get('refresh_token'))
        self._config['gmail']['token_json'] = os.getenv("GMAIL_TOKEN_JSON", self._config['gmail'].get('token_json'))

        self._config['ml'] = self._config.get('ml', {})
        self._config['ml']['model_path'] = os.getenv("MODEL_PATH", self._config['ml'].get('model_path', "models/model.pth"))

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value by key."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

def get_config() -> ConfigLoader:
    """Returns the singleton ConfigLoader instance."""
    return ConfigLoader()
