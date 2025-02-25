import logging

logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(levelname)s - %(message)s",  
)

# Create a global logger instance
logger = logging.getLogger("app_logger")
