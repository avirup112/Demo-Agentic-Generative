"""Configuration settings for the RAG Q&A Agent."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the RAG Q&A Agent."""
    
    # LLM Configuration
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    
    # Model Settings
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Tracing and Evaluation
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "rag-qa-agent")
    ENABLE_TRACING: bool = os.getenv("ENABLE_TRACING", "false").lower() == "true"
    ENABLE_EVALUATION: bool = os.getenv("ENABLE_EVALUATION", "true").lower() == "true"
    
    # Vector Database Settings
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "knowledge_base")
    
    # Retrieval Settings
    TOP_K_DOCUMENTS: int = int(os.getenv("TOP_K_DOCUMENTS", "3"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Document Processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Knowledge Base
    KNOWLEDGE_BASE_DIR: str = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # UI Settings
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # Evaluation Settings
    EVALUATION_DATASET_PATH: str = os.getenv("EVALUATION_DATASET_PATH", "./evaluation_dataset.json")
    GROUND_TRUTH_PATH: str = os.getenv("GROUND_TRUTH_PATH", "./ground_truth.json")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings."""
        groq_models = [
            "llama-3.1-8b-instant", "llama-3.1-70b-versatile", 
            "llama-3.2-1b-preview", "llama-3.2-3b-preview",
            "mixtral-8x7b-32768", "gemma2-9b-it"
        ]
        
        if not cls.GROQ_API_KEY and cls.LLM_MODEL in groq_models:
            print("Warning: Groq API key not found. Please set GROQ_API_KEY environment variable.")
            print("Get your API key from: https://console.groq.com/")
            return False
        return True