import os
import logging
from datetime import datetime

# --- 1. Advanced Logging Configuration ---
# This helps you debug the 'Why' and 'How' if something fails on your M3
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_debug.log"), # Saves errors to a file
        logging.StreamHandler()                # Prints errors to your terminal
    ]
)
logger = logging.getLogger("EnterpriseTutor")

def validate_env():
    """
    The 'Why': Large projects crash if API keys are missing. 
    This check prevents the app from starting without the Gemini Key.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Missing GOOGLE_API_KEY in .env file!")
        return False
    return True

def get_timestamp():
    """Useful for versioning your vector database backups."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def clean_metadata(metadata: dict):
    """
    The 'Why': Some PDF loaders add messy metadata that Qdrant doesn't like. 
    This ensures our data is 'Enterprise Clean'.
    """
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value) # Convert lists/dicts to strings
    return cleaned

def ensure_dirs():
    """Creates necessary folders if they don't exist."""
    dirs = ["data/docs", "vector_db", "logs"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info(f"Created directory: {d}")