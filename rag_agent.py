"""Main RAG Q&A Agent implementation using LangGraph."""

import logging
import time
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from models import AgentState, QueryResult, Document, WorkflowError
from workflow_nodes import WorkflowNodes
from vector_store import VectorStore
from llm_service import LLMService
from document_processor import DocumentProcessor
from config import Config
from tracing import tracing_manager, trace_rag_query, TracingSession

logger = logging.getLogger(__name__)

class RAGAgent:
    """Main RAG Q&A Agent using LangGraph workflow."""
    
    def __init__(self, initialize_kb: bool = True):
        """Initialize the RAG Agent."""
        logger.info("Initializing RAG Agent...")
        
        # Initialize services
        self.vector_store = VectorStore()
        self.llm_service = LLMService()
        self.document_processor = DocumentProcessor()
        self.workflow_nodes = WorkflowNodes(self.vector_store, self.llm_service)
        
        # Initialize knowledge base if requested
        if initialize_kb:
            self._initialize_knowledge_base()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("RAG Agent initialized successfully")
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with documents."""
        try:
            # Check if knowledge base is already populated
            collection_info = self.vector_store.get_collection_info()
            if collection_info.get('document_count', 0) > 0:
                logger.info(f"Knowledge base already contains {collection_info['document_count']} documents")
                return
            
            logger.info("Loading documents into knowledge base...")
            
            # Load documents from knowledge base directory
            documents = self.document_processor.load_documents_from_directory(
                Config.KNOWLEDGE_BASE_DIR
            )
            
            if not documents:
                logger.warning("No documents found in knowledge base directory")
                return
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            # Log statistics
            stats = self.document_processor.get_document_stats(documents)
            logger.info(f"Knowledge base initialized with {stats['total_documents']} document chunks")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            raise WorkflowError(f"Knowledge base initialization failed: {e}")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        logger.info("Building LangGraph workflow...")
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan", self.workflow_nodes.plan_node)
        workflow.add_node("retrieve", self.workflow_nodes.retrieve_node)
        workflow.add_node("answer", self.workflow_nodes.answer_node)
        workflow.add_node("reflect", self.workflow_nodes.reflect_node)
        
        # Define the workflow edges
        workflow.set_entry_point("plan")
        
        # Plan -> Retrieve (always go to retrieve, it will handle no_retrieval case)
        workflow.add_edge("plan", "retrieve")
        
        # Retrieve -> Answer
        workflow.add_edge("retrieve", "answer")
        
        # Answer -> Reflect
        workflow.add_edge("answer", "reflect")
        
        # Reflect -> END
        workflow.add_edge("reflect", END)
        
        # Compile the workflow
        memory = MemorySaver()
        compiled_workflow = workflow.compile(checkpointer=memory)
        
        logger.info("LangGraph workflow built successfully")
        return compiled_workflow
    
    @trace_rag_query
    def query(self, question: str, session_id: str = "default") -> QueryResult:
        """Process a query through the RAG workflow."""
        logger.info(f"Processing query: '{question[:50]}...'")
        
        with TracingSession(f"query_{session_id}"):
            try:
                # Initialize state
                initial_state: AgentState = {
                    "query": question,
                    "plan_decision": "",
                    "retrieved_docs": [],
                    "answer": "",
                    "reflection": "",
                    "confidence_score": 0.0,
                    "processing_steps": [],
                    "start_time": time.time()
                }
            
                # Execute workflow
                config = {"configurable": {"thread_id": session_id}}
                final_state = self.workflow.invoke(initial_state, config)
                
                # Calculate processing time
                processing_time = time.time() - initial_state["start_time"]
                
                # Create result object
                result = QueryResult(
                    original_query=question,
                    processed_query=question,  # Could be modified in future versions
                    retrieved_docs=self._parse_retrieved_docs(final_state.get("retrieved_docs", [])),
                    answer=final_state.get("answer", ""),
                    confidence=final_state.get("confidence_score", 0.0),
                    reflection=final_state.get("reflection", ""),
                    processing_time=processing_time,
                    processing_steps=final_state.get("processing_steps", [])
                )
                
                logger.info(f"Query processed successfully in {processing_time:.2f} seconds")
                
                # Log to tracing system
                tracing_manager.log_answer_generation(
                    query=question,
                    context="\n".join(final_state.get("retrieved_docs", [])),
                    answer=result.answer,
                    confidence=result.confidence
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Query processing failed: {e}")
                raise WorkflowError(f"Query processing failed: {e}")
    
    def _parse_retrieved_docs(self, retrieved_docs: List[str]) -> List[Document]:
        """Parse retrieved document strings back into Document objects."""
        documents = []
        
        for doc_text in retrieved_docs:
            # Extract metadata from the formatted string
            lines = doc_text.split('\n', 1)
            if len(lines) >= 2 and lines[0].startswith('[Document'):
                # Parse the header line for score
                header = lines[0]
                content = lines[1]
                
                # Extract score if available
                score = 0.0
                if 'Score:' in header:
                    try:
                        score_part = header.split('Score:')[1].split(']')[0].strip()
                        score = float(score_part)
                    except (ValueError, IndexError):
                        pass
                
                document = Document(
                    content=content,
                    metadata={'retrieval_score': score},
                    source="retrieved"
                )
                documents.append(document)
            else:
                # Fallback for documents without proper formatting
                document = Document(
                    content=doc_text,
                    metadata={},
                    source="retrieved"
                )
                documents.append(document)
        
        return documents
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the knowledge base."""
        try:
            collection_info = self.vector_store.get_collection_info()
            return {
                'status': 'ready',
                'collection_info': collection_info,
                'config': {
                    'chunk_size': Config.CHUNK_SIZE,
                    'chunk_overlap': Config.CHUNK_OVERLAP,
                    'top_k_documents': Config.TOP_K_DOCUMENTS,
                    'similarity_threshold': Config.SIMILARITY_THRESHOLD,
                    'embedding_model': Config.EMBEDDING_MODEL,
                    'llm_model': Config.LLM_MODEL
                }
            }
        except Exception as e:
            logger.error(f"Failed to get knowledge base info: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def add_documents_to_kb(self, documents: List[Document]) -> bool:
        """Add new documents to the knowledge base."""
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to knowledge base")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def clear_knowledge_base(self) -> bool:
        """Clear all documents from the knowledge base."""
        try:
            self.vector_store.clear_collection()
            logger.info("Knowledge base cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear knowledge base: {e}")
            return False
    
    def batch_query(self, questions: List[str], session_id: str = "batch") -> List[QueryResult]:
        """Process multiple queries in batch."""
        logger.info(f"Processing batch of {len(questions)} queries")
        
        results = []
        for i, question in enumerate(questions):
            try:
                result = self.query(question, f"{session_id}_{i}")
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process query {i+1}: {e}")
                # Create error result
                error_result = QueryResult(
                    original_query=question,
                    processed_query=question,
                    retrieved_docs=[],
                    answer=f"Error processing query: {str(e)}",
                    confidence=0.0,
                    reflection="Error occurred during processing",
                    processing_time=0.0,
                    processing_steps=[f"Error: {str(e)}"]
                )
                results.append(error_result)
        
        logger.info(f"Batch processing completed: {len(results)} results")
        return results

# Utility functions for testing
def test_rag_agent():
    """Test the complete RAG Agent."""
    print("Testing RAG Agent...")
    
    # Initialize agent
    agent = RAGAgent(initialize_kb=True)
    
    # Test queries
    test_queries = [
        "What are the benefits of renewable energy?",
        "How does machine learning work?",
        "What causes climate change?",
        "Hello, how are you?",  # Should not need retrieval
        "Explain artificial intelligence applications"
    ]
    
    print(f"\nTesting {len(test_queries)} queries:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        try:
            result = agent.query(query)
            
            print(f"Answer: {result.answer}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Processing time: {result.processing_time:.2f}s")
            print(f"Retrieved docs: {len(result.retrieved_docs)}")
            
            if result.reflection:
                print(f"Reflection: {result.reflection[:100]}...")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Test knowledge base info
    print("\n--- Knowledge Base Info ---")
    kb_info = agent.get_knowledge_base_info()
    print(f"Status: {kb_info.get('status')}")
    if 'collection_info' in kb_info:
        collection_info = kb_info['collection_info']
        print(f"Documents: {collection_info.get('document_count', 0)}")
        print(f"Embedding model: {collection_info.get('embedding_model', 'unknown')}")
    
    print("RAG Agent test completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    test_rag_agent()