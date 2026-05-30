import os  
from dotenv import load_dotenv

load_dotenv()

AZURE_ENDPOINT    = os.getenv("AZURE_ENDPOINT", "https://az-oai-collection.openai.azure.com/")  
AZURE_API_KEY     = os.getenv("AZURE_API_KEY", "3rPjrg7Znwpbh9BBTc86Lr4S0gJHYWc7AV95rGpHbBWpW9BkpBLjJQQJ99CBAC5RqLJXJ3w3AAABACOG4GzS")  
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")  
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  
CHAT_MODEL        = os.getenv("CHAT_MODEL", "gpt-4.1")  
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", 500))  
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", 50))  
TOP_K             = int(os.getenv("TOP_K", 5))

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))  
UPLOAD_DIR   = os.path.join(BASE_DIR, "storage", "uploads")  
CHROMA_DIR   = os.path.join(BASE_DIR, "storage", "chroma_db")  
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")  
DOC_RELEVANCE_THRESHOLD   = float(os.getenv("DOC_RELEVANCE_THRESHOLD", 0.25))  
CHUNK_RELEVANCE_THRESHOLD = float(os.getenv("CHUNK_RELEVANCE_THRESHOLD", 0.20))  
