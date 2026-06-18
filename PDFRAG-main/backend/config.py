import os  
from dotenv import load_dotenv

load_dotenv()

AZURE_ENDPOINT    = os.getenv("AZURE_ENDPOINT")  
AZURE_API_KEY     = os.getenv("AZURE_API_KEY")  
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")  
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL")  
CHAT_MODEL        = os.getenv("CHAT_MODEL")  
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", 500))  
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", 50))  
TOP_K             = int(os.getenv("TOP_K", 5))

# PDF Processing
PDF_EXTRACT_TABLES = os.getenv("PDF_EXTRACT_TABLES", "false").lower() == "true"  # Disable table extraction for speed

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))  
UPLOAD_DIR   = os.path.join(BASE_DIR, "storage", "uploads")  
CHROMA_DIR   = os.path.join(BASE_DIR, "storage", "chroma_db")  
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")  
DOC_RELEVANCE_THRESHOLD   = float(os.getenv("DOC_RELEVANCE_THRESHOLD", 0.25))  
CHUNK_RELEVANCE_THRESHOLD = float(os.getenv("CHUNK_RELEVANCE_THRESHOLD", 0.20))  
