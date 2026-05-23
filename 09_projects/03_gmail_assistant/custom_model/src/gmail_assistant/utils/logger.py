import logging
import logging.config
import os
import yaml
from typing import Optional

def setup_logging(
    default_path: str = 'config/logging_config.yaml',
    default_level: int = logging.INFO,
    env_key: str = 'LOG_CFG'
):
    """
    Setup logging configuration from a YAML file or use default settings.
    
    Args:
        default_path: Path to the logging configuration YAML file.
        default_level: Default logging level if config file is not found.
        env_key: Environment variable to override the config file path.
    """
    path = os.getenv(env_key, default_path)
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print(f"Error in Logging Configuration: {e}")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the specified name."""
    return logging.getLogger(name)
