#!/usr/bin/env python3
"""
PDFRAG - Unified Entry Point
Starts both backend (port 8000) and frontend (port 3000) servers

Usage:
    python run.py              # Always clears vector store & memory on startup
    python run.py --fresh      # Same behavior (kept for compatibility)
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
    """Start backend API server on port 8000 (FastAPI + uvicorn)."""
    print("\n" + "="*60)
    print("🚀 Starting BACKEND (FastAPI/uvicorn, port 8000)...")
    print("="*60 + "\n")

    os.chdir(str(PROJECT_ROOT / "PDFRAG-main"))
    # Launch the FastAPI app via uvicorn. The module path is backend.app:app.
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.app:app",
        "--host", "localhost",
        "--port", "8000",
        "--log-level", "warning",
    ])

def run_frontend():
    """Start frontend HTTP server on port 3000"""
    print("\n" + "="*60)
    print("🚀 Starting FRONTEND (port 3000)...")
    print("="*60 + "\n")
    
    os.chdir(str(FRONTEND_PATH))
    subprocess.run([sys.executable, "-m", "http.server", "3000"])

def wait_for_backend(host="localhost", port=8000, timeout=90):
    """Block until the backend accepts TCP connections, or until ``timeout`` seconds.

    uvicorn binds the port only AFTER it imports the app, which runs all the heavy
    pipeline singleton initialization (Embedder, VectorStore, memory manager, ...). So a
    successful connection means the backend is fully loaded and ready to serve. Polling
    for this — instead of a blind fixed sleep — ensures the frontend (and the browser)
    only come up once queries will actually work. Returns True if ready, False on timeout.
    """
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False

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
    # Always run with fresh data as requested.
    # Keep --fresh accepted for backward compatibility.
    fresh_start = True
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
            
            # Wait until the backend is actually ready before starting the frontend.
            # uvicorn binds port 8000 only AFTER importing the app (which initializes every
            # pipeline singleton), so a successful connection means "fully loaded". This
            # replaces a blind fixed sleep, so the frontend/browser never come up while the
            # backend is still initializing.
            print("⏳ Waiting for backend to finish initializing...")
            if wait_for_backend(port=8000, timeout=90):
                print("✅ Backend ready\n")
            else:
                print("⚠️  Backend not ready after 90s — starting frontend anyway\n")
            
            frontend_thread = threading.Thread(target=run_frontend, daemon=False)
            frontend_thread.start()
            
            print("\n" + "="*60)
            print("✅ Both servers running!")
            print("="*60)
            print("📄 Backend:  http://localhost:8000")
            print("🌐 Frontend: http://localhost:3000")
            print("\n⏹️  Press Ctrl+C to stop both servers")
            print("📝 Startup mode: always fresh (storage cleared on launch)\n")
            
            # Keep main thread alive
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Shutting down servers...")
        sys.exit(0)

if __name__ == "__main__":
    main()
