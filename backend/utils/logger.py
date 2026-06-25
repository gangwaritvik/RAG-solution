import logging  
import os  
import sys
from datetime import datetime

# ── Create logs directory ──  
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")  
os.makedirs(LOG_DIR, exist_ok=True)

# ── Log file with timestamp ──  
log_file = os.path.join(LOG_DIR, f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# ── Custom StreamHandler with flush and UTF-8 encoding ──
class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Encode to UTF-8 and write to stdout to handle emoji properly
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout.buffer.write((msg + self.terminator).encode('utf-8'))
            else:
                self.stream.write(msg + self.terminator)
            self.stream.flush()  # Force flush after every log
            sys.stdout.flush()  # Also flush stdout
        except Exception:
            self.handleError(record)

# ── Configure logger ──  
logging.basicConfig(  
    level=logging.DEBUG,  
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",  
    datefmt="%Y-%m-%d %H:%M:%S",  
    handlers=[  
        logging.FileHandler(log_file, encoding="utf-8"),  
        FlushStreamHandler(sys.stdout),   # also prints to terminal with explicit flush
    ]  
)

# ── Silence noisy third-party loggers ──
# These log full HTTP request/response bodies at DEBUG level (including every
# retrieved chunk sent to the LLM), which makes our logs unreadable. Raise their
# level to WARNING so only real problems show up, while keeping our own DEBUG logs.
for _noisy in ("openai", "httpx", "httpcore", "urllib3", "chromadb", "azure"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:  
    return logging.getLogger(name)  
