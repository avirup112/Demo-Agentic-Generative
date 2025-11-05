"""Data models and interfaces for the RAG Q&A Agent."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
import time

class AgentState(TypedDict):
    """State management for the LangGraph workflow."""
    query: str
    plan_decision: str
    retrieved_docs: List[str]
    answer: str
    reflection: str
    confidence_score: float
    processing_steps: List[str]
    start_time: float

@dataclass
class Document:
    """Document model for knowledge base entries."""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    source: str = ""
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if not self.metadata:
            self.metadata = {}
        if 'id' not in self.metadata:
            self.metadata['id'] = hash(self.content[:100])

@dataclass
class QueryResult:
    """Complete result of query processing."""
    original_query: str
    processed_query: str
    retrieved_docs: List[Document]
    answer: str
    confidence: float
    reflection: str
    processing_time: float
    processing_steps: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'original_query': self.original_query,
            'processed_query': self.processed_query,
            'retrieved_docs': [
                {
                    'content': doc.content[:200] + '...' if len(doc.content) > 200 else doc.content,
                    'source': doc.source,
                    'metadata': doc.metadata
                } for doc in self.retrieved_docs
            ],
            'answer': self.answer,
            'confidence': self.confidence,
            'reflection': self.reflection,
            'processing_time': self.processing_time,
            'processing_steps': self.processing_steps
        }

@dataclass
class RetrievalResult:
    """Result of document retrieval operation."""
    documents: List[Document]
    scores: List[float]
    query: str
    total_docs_searched: int
    
    def get_top_k(self, k: int) -> List[Document]:
        """Get top k documents by relevance score."""
        sorted_docs = sorted(
            zip(self.documents, self.scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        return [doc for doc, _ in sorted_docs[:k]]

@dataclass
class ReflectionResult:
    """Result of answer reflection and evaluation."""
    is_relevant: bool
    confidence_score: float
    reasoning: str
    suggestions: List[str]
    answer_quality: str  # 'excellent', 'good', 'fair', 'poor'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'is_relevant': self.is_relevant,
            'confidence_score': self.confidence_score,
            'reasoning': self.reasoning,
            'suggestions': self.suggestions,
            'answer_quality': self.answer_quality
        }

class ProcessingStep:
    """Utility class for tracking processing steps."""
    
    @staticmethod
    def create_step(step_name: str, details: str, timestamp: Optional[float] = None) -> str:
        """Create a formatted processing step."""
        if timestamp is None:
            timestamp = time.time()
        return f"[{timestamp:.2f}] {step_name}: {details}"
    
    @staticmethod
    def log_step(state: AgentState, step_name: str, details: str) -> None:
        """Add a processing step to the agent state."""
        step = ProcessingStep.create_step(step_name, details)
        if 'processing_steps' not in state:
            state['processing_steps'] = []
        state['processing_steps'].append(step)
        print(f"🔄 {step}")

# Error classes for better error handling
class RAGAgentError(Exception):
    """Base exception for RAG Agent errors."""
    pass

class VectorStoreError(RAGAgentError):
    """Exception for vector store operations."""
    pass

class LLMServiceError(RAGAgentError):
    """Exception for LLM service operations."""
    pass

class DocumentProcessingError(RAGAgentError):
    """Exception for document processing operations."""
    pass

class WorkflowError(RAGAgentError):
    """Exception for workflow execution errors."""
    pass