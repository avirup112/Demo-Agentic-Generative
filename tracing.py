"""LangSmith tracing integration for RAG Q&A Agent."""

import os
import time
import logging
from typing import Dict, Any, Optional
from functools import wraps

try:
    from langsmith import Client
    from langsmith.run_helpers import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logging.info("LangSmith not available. This is optional for basic functionality.")

try:
    from trulens_eval import TruChain, Feedback, Tru
    from trulens_eval.feedback import Groundedness
    from trulens_eval.feedback.provider.openai import OpenAI as TruOpenAI
    TRULENS_AVAILABLE = True
except ImportError:
    TRULENS_AVAILABLE = False
    logging.info("TruLens not available. This is optional for basic functionality.")

from config import Config

logger = logging.getLogger(__name__)

class TracingManager:
    """Manages tracing and evaluation logging for the RAG system."""
    
    def __init__(self):
        self.langsmith_client = None
        self.trulens_session = None
        
        # Initialize LangSmith if available and configured
        if LANGSMITH_AVAILABLE and Config.LANGSMITH_API_KEY and Config.ENABLE_TRACING:
            try:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
                os.environ["LANGCHAIN_API_KEY"] = Config.LANGSMITH_API_KEY
                os.environ["LANGCHAIN_PROJECT"] = Config.LANGSMITH_PROJECT
                
                self.langsmith_client = Client(api_key=Config.LANGSMITH_API_KEY)
                logger.info("LangSmith tracing initialized")
            except Exception as e:
                logger.error(f"Failed to initialize LangSmith: {e}")
        
        # Initialize TruLens if available
        if TRULENS_AVAILABLE and Config.ENABLE_TRACING:
            try:
                self.tru = Tru()
                logger.info("TruLens evaluation initialized")
            except Exception as e:
                logger.error(f"Failed to initialize TruLens: {e}")
    
    def trace_query(self, func):
        """Decorator to trace query processing."""
        if not LANGSMITH_AVAILABLE or not self.langsmith_client:
            return func
        
        @traceable(name="rag_query_processing")
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    
    def trace_node_execution(self, node_name: str):
        """Decorator to trace individual node execution."""
        def decorator(func):
            if not LANGSMITH_AVAILABLE or not self.langsmith_client:
                return func
            
            @traceable(name=f"rag_node_{node_name}")
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def log_retrieval_context(self, query: str, retrieved_docs: list, scores: list):
        """Log retrieval context for analysis."""
        if not LANGSMITH_AVAILABLE or not self.langsmith_client:
            return
        
        try:
            context_data = {
                "query": query,
                "num_retrieved": len(retrieved_docs),
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "top_score": max(scores) if scores else 0,
                "documents": [
                    {
                        "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        "score": score,
                        "source": getattr(doc, 'source', 'unknown')
                    }
                    for doc, score in zip(retrieved_docs, scores)
                ]
            }
            
            # Log to LangSmith
            self.langsmith_client.create_run(
                name="retrieval_context",
                run_type="retriever",
                inputs={"query": query},
                outputs=context_data
            )
            
        except Exception as e:
            logger.error(f"Failed to log retrieval context: {e}")
    
    def log_answer_generation(self, query: str, context: str, answer: str, confidence: float):
        """Log answer generation for analysis."""
        if not LANGSMITH_AVAILABLE or not self.langsmith_client:
            return
        
        try:
            generation_data = {
                "query": query,
                "context_length": len(context),
                "answer": answer,
                "confidence": confidence,
                "answer_length": len(answer)
            }
            
            # Log to LangSmith
            self.langsmith_client.create_run(
                name="answer_generation",
                run_type="llm",
                inputs={"query": query, "context": context},
                outputs=generation_data
            )
            
        except Exception as e:
            logger.error(f"Failed to log answer generation: {e}")
    
    def create_trulens_app(self, rag_agent):
        """Create TruLens application wrapper for evaluation."""
        if not TRULENS_AVAILABLE or not self.tru:
            return None
        
        try:
            # Define feedback functions
            feedback_functions = []
            
            # Groundedness feedback (requires OpenAI)
            if Config.GROQ_API_KEY:  # Using Groq as proxy for LLM availability
                try:
                    # Note: TruLens typically expects OpenAI, but we can adapt
                    groundedness = Groundedness()
                    
                    f_groundedness = Feedback(
                        groundedness.groundedness_measure_with_cot_reasons,
                        name="Groundedness"
                    ).on_input_output()
                    
                    feedback_functions.append(f_groundedness)
                except Exception as e:
                    logger.warning(f"Could not create groundedness feedback: {e}")
            
            # Create TruChain wrapper
            tru_rag = TruChain(
                rag_agent,
                app_id="rag_qa_agent",
                feedbacks=feedback_functions
            )
            
            return tru_rag
            
        except Exception as e:
            logger.error(f"Failed to create TruLens app: {e}")
            return None
    
    def log_evaluation_metrics(self, query: str, answer: str, metrics: Dict[str, Any]):
        """Log evaluation metrics to tracing systems."""
        if not LANGSMITH_AVAILABLE or not self.langsmith_client:
            return
        
        try:
            # Log to LangSmith
            self.langsmith_client.create_run(
                name="evaluation_metrics",
                run_type="tool",
                inputs={"query": query, "answer": answer},
                outputs=metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to log evaluation metrics: {e}")
    
    def get_tracing_stats(self) -> Dict[str, Any]:
        """Get tracing statistics and insights."""
        stats = {
            "langsmith_enabled": LANGSMITH_AVAILABLE and self.langsmith_client is not None,
            "trulens_enabled": TRULENS_AVAILABLE and self.trulens_session is not None,
            "project_name": Config.LANGSMITH_PROJECT if Config.LANGSMITH_PROJECT else "Not configured"
        }
        
        if LANGSMITH_AVAILABLE and self.langsmith_client:
            try:
                # Get recent runs (this would need proper API calls)
                stats["recent_runs"] = "Available via LangSmith dashboard"
            except Exception as e:
                stats["langsmith_error"] = str(e)
        
        return stats

# Global tracing manager instance
tracing_manager = TracingManager()

# Decorator functions for easy use
def trace_rag_query(func):
    """Decorator to trace RAG query processing."""
    return tracing_manager.trace_query(func)

def trace_node(node_name: str):
    """Decorator to trace node execution."""
    return tracing_manager.trace_node_execution(node_name)

# Context manager for tracing sessions
class TracingSession:
    """Context manager for tracing sessions."""
    
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting tracing session: {self.session_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Tracing session '{self.session_name}' completed in {duration:.2f}s")
        
        if exc_type:
            logger.error(f"Tracing session '{self.session_name}' failed: {exc_val}")

# Utility functions
def log_rag_execution(query: str, result_data: Dict[str, Any]):
    """Log complete RAG execution."""
    tracing_manager.log_answer_generation(
        query=query,
        context=result_data.get('context', ''),
        answer=result_data.get('answer', ''),
        confidence=result_data.get('confidence', 0.0)
    )

def setup_tracing():
    """Setup tracing configuration."""
    if Config.ENABLE_TRACING:
        logger.info("Tracing enabled")
        
        if Config.LANGSMITH_API_KEY:
            logger.info(f"LangSmith project: {Config.LANGSMITH_PROJECT}")
        else:
            logger.warning("LangSmith API key not configured")
    else:
        logger.info("Tracing disabled")

# Example usage and testing
def test_tracing():
    """Test tracing functionality."""
    print("🔍 Testing Tracing Integration...")
    
    # Test tracing manager initialization
    print(f"LangSmith available: {LANGSMITH_AVAILABLE}")
    print(f"TruLens available: {TRULENS_AVAILABLE}")
    
    # Get tracing stats
    stats = tracing_manager.get_tracing_stats()
    print(f"Tracing stats: {stats}")
    
    # Test tracing session
    with TracingSession("test_session"):
        print("Executing traced operations...")
        
        # Simulate logging
        tracing_manager.log_answer_generation(
            query="Test query",
            context="Test context",
            answer="Test answer",
            confidence=0.85
        )
    
    print("✅ Tracing test completed!")

if __name__ == "__main__":
    import time
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Setup and test tracing
    setup_tracing()
    test_tracing()