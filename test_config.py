import os
import sys
sys.path.insert(0, 'PDFRAG-main')
from backend import config

print(f"PDF_EXTRACT_TABLES = {config.PDF_EXTRACT_TABLES}")
print(f"Environment variable: {os.getenv('PDF_EXTRACT_TABLES', 'NOT SET')}")
