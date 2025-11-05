"""Simple launcher for the RAG Q&A Agent."""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app."""
    print("🤖 Starting RAG Q&A Agent...")
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0'
        ])
    except KeyboardInterrupt:
        print("\n👋 RAG Q&A Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()