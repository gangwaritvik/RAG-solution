import logging  
import os  
from datetime import datetime

# ── Create logs directory ──  
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")  
os.makedirs(LOG_DIR, exist_ok=True)

# ── Log file with timestamp ──  
log_file = os.path.join(LOG_DIR, f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# ── Configure logger ──  
logging.basicConfig(  
    level=logging.DEBUG,  
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",  
    datefmt="%Y-%m-%d %H:%M:%S",  
    handlers=[  
        logging.FileHandler(log_file, encoding="utf-8"),  
        logging.StreamHandler(),   # also prints to terminal  
    ]  
)

def get_logger(name: str) -> logging.Logger:  
    return logging.getLogger(name)  
