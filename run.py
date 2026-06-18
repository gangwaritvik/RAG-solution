#!/usr/bin/env python3
"""
PDFRAG - Unified Entry Point
Starts both backend (port 8000) and frontend (port 3000) servers

Usage:
    python run.py              # Normal run with persistent data
    python run.py --fresh      # Clear vector store & memory on startup
    python run.py --backend    # Backend only
    python run.py --frontend   # Frontend only
"""

import os
import sys
import subprocess
import time
import threading
import shutil
from pathlib import Path

# Fix Windows Unicode encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Get project root
PROJECT_ROOT = Path(__file__).parent.absolute()
BACKEND_PATH = PROJECT_ROOT / "PDFRAG-main" / "backend" / "main.py"
FRONTEND_PATH = PROJECT_ROOT / "PDFRAG-main" / "frontend"
STORAGE_PATH = PROJECT_ROOT / "PDFRAG-main" / "backend" / "storage"
CHROMA_PATH = STORAGE_PATH / "chroma_db"

def run_backend():
    """Start backend HTTP server on port 8000"""
    print("\n" + "="*60)
    print("🚀 Starting BACKEND (port 8000)...")
    print("="*60 + "\n")
    
    os.chdir(str(PROJECT_ROOT))
    subprocess.run([sys.executable, str(BACKEND_PATH)])

def run_frontend():
    """Start frontend HTTP server on port 3000"""
    print("\n" + "="*60)
    print("🚀 Starting FRONTEND (port 3000)...")
    print("="*60 + "\n")
    
    os.chdir(str(FRONTEND_PATH))
    subprocess.run([sys.executable, "-m", "http.server", "3000"])

def clear_storage():
    """Clear all persisted data (vectors, conversations, memory)"""
    import time
    
    try:
        # Clear ChromaDB
        if CHROMA_PATH.exists():
            print(f"🗑️  Clearing vector store: {CHROMA_PATH}")
            try:
                shutil.rmtree(CHROMA_PATH)
                print("✅ Vector store cleared")
            except PermissionError:
                print("⚠️  Files locked (backend may still be running)")
                print("   Stop the server first: Ctrl+C")
                sys.exit(1)
        
        # Clear uploads
        uploads_path = STORAGE_PATH / "uploads"
        if uploads_path.exists():
            print(f"🗑️  Clearing uploads: {uploads_path}")
            for file in uploads_path.glob("*"):
                try:
                    if file.is_file():
                        file.unlink()
                except PermissionError:
                    pass  # File may be locked, continue
            print("✅ Uploads cleared")
            
    except Exception as e:
        print(f"❌ Error clearing storage: {e}")
        sys.exit(1)

def main():
    # Parse command-line arguments
    fresh_start = "--fresh" in sys.argv
    backend_only = "--backend" in sys.argv
    frontend_only = "--frontend" in sys.argv
    
    print("\n")
    print("╔" + "─"*58 + "╗")
    print("║  ⬡ PDF RAG — Enterprise RAG System                     ║")
    if fresh_start:
        print("║  Starting with FRESH DATA (clearing vector store)      ║")
    else:
        print("║  Starting servers...                                   ║")
    print("╚" + "─"*58 + "╝")
    print()
    
    # Verify files exist
    if not BACKEND_PATH.exists():
        print(f"❌ Backend not found: {BACKEND_PATH}")
        sys.exit(1)
    
    if not FRONTEND_PATH.exists():
        print(f"❌ Frontend directory not found: {FRONTEND_PATH}")
        sys.exit(1)
    
    # **IMPORTANT**: Clear storage BEFORE starting backend
    # This prevents file lock conflicts with ChromaDB
    if fresh_start:
        print("🗑️  Clearing storage before startup...")
        clear_storage()
        print()
    
    # Start services based on flags
    try:
        if frontend_only:
            print("🌐 Frontend only mode\n")
            run_frontend()
        elif backend_only:
            print("🔧 Backend only mode\n")
            run_backend()
        else:
            # Start both servers
            backend_thread = threading.Thread(target=run_backend, daemon=False)
            backend_thread.start()
            
            # Give backend time to start
            time.sleep(3)
            
            frontend_thread = threading.Thread(target=run_frontend, daemon=False)
            frontend_thread.start()
            
            print("\n" + "="*60)
            print("✅ Both servers running!")
            print("="*60)
            print("📄 Backend:  http://localhost:8000")
            print("🌐 Frontend: http://localhost:3000")
            print("\n⏹️  Press Ctrl+C to stop both servers")
            print("📝 Tip: Use 'python run.py --fresh' to start with clean data\n")
            
            # Keep main thread alive
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Shutting down servers...")
        sys.exit(0)

if __name__ == "__main__":
    main()
