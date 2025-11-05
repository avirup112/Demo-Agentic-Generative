"""LangGraph workflow nodes for the RAG Q&A Agent."""

import logging
import time
from typing import Dict, Any, List

from models import AgentState, ProcessingStep, WorkflowError
from vector_store import VectorStore
from llm_service import LLMService
from config import Config
from tracing import trace_node, tracing_manager

logger = logging.getLogger(__name__)

class WorkflowNodes:
    """Container for all LangGraph workflow nodes."""
    
    def __init__(self, vector_store: VectorStore = None, llm_service: LLMService = None):
        """Initialize workflow nodes with required services."""
        self.vector_store = vector_store or VectorStore()
        self.llm_service = llm_service or LLMService()
        
        logger.info("Initialized workflow nodes")
    
    @trace_node("plan")
    def plan_node(self, state: AgentState) -> AgentState:
        """
        Plan node: Analyze the query and decide if retrieval is needed.
        
        This node examines the user's query to determine:
        1. Whether the query requires knowledge base lookup
        2. How to process the query for optimal retrieval
        3. What type of information is being requested
        """
        try:
            query = state["query"]
            ProcessingStep.log_step(state, "PLAN", f"Analyzing query: '{query[:50]}...'")
            
            # Simple heuristic-based planning
            plan_decision = self._analyze_query_for_retrieval(query)
            
            # Update state
            state["plan_decision"] = plan_decision
            
            ProcessingStep.log_step(
                state, 
                "PLAN_RESULT", 
                f"Decision: {plan_decision}"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Plan node failed: {e}")
            raise WorkflowError(f"Plan node execution failed: {e}")
    
    @trace_node("retrieve")
    def retrieve_node(self, state: AgentState) -> AgentState:
        """
        Retrieve node: Perform RAG using vector database.
        
        This node:
        1. Uses the query to search the vector database
        2. Retrieves the most relevant document chunks
        3. Formats the context for the answer generation
        """
        try:
            query = state["query"]
            plan_decision = state.get("plan_decision", "retrieve")
            
            ProcessingStep.log_step(state, "RETRIEVE", f"Searching knowledge base for: '{query[:50]}...'")
            
            if plan_decision == "no_retrieval":
                # Skip retrieval if not needed
                state["retrieved_docs"] = []
                ProcessingStep.log_step(state, "RETRIEVE_SKIP", "No retrieval needed based on plan")
                return state
            
            # Perform vector similarity search
            retrieval_result = self.vector_store.similarity_search(
                query=query,
                k=Config.TOP_K_DOCUMENTS
            )
            
            # Format retrieved documents
            retrieved_docs = []
            for i, (doc, score) in enumerate(zip(retrieval_result.documents, retrieval_result.scores)):
                if score >= Config.SIMILARITY_THRESHOLD:
                    doc_text = f"[Document {i+1} - Score: {score:.3f}]\n{doc.content}\n"
                    retrieved_docs.append(doc_text)
            
            state["retrieved_docs"] = retrieved_docs
            
            ProcessingStep.log_step(
                state, 
                "RETRIEVE_RESULT", 
                f"Retrieved {len(retrieved_docs)} relevant documents"
            )
            
            # Log document details
            for i, doc in enumerate(retrieval_result.documents[:3]):  # Log first 3
                ProcessingStep.log_step(
                    state,
                    f"DOC_{i+1}",
                    f"Source: {doc.source}, Score: {retrieval_result.scores[i]:.3f}, Preview: {doc.content[:100]}..."
                )
            
            return state
            
        except Exception as e:
            logger.error(f"Retrieve node failed: {e}")
            raise WorkflowError(f"Retrieve node execution failed: {e}")
    
    @trace_node("answer")
    def answer_node(self, state: AgentState) -> AgentState:
        """
        Answer node: Generate response using LLM and retrieved context.
        
        This node:
        1. Combines the retrieved documents into context
        2. Uses the LLM to generate an answer
        3. Handles cases where no relevant context is found
        """
        try:
            query = state["query"]
            retrieved_docs = state.get("retrieved_docs", [])
            
            ProcessingStep.log_step(state, "ANSWER", f"Generating answer for: '{query[:50]}...'")
            
            # Prepare context from retrieved documents
            if retrieved_docs:
                context = "\n\n".join(retrieved_docs)
                ProcessingStep.log_step(
                    state, 
                    "CONTEXT", 
                    f"Using {len(retrieved_docs)} documents as context ({len(context)} chars)"
                )
            else:
                context = "No relevant documents found in the knowledge base."
                ProcessingStep.log_step(state, "CONTEXT", "No context available - generating general response")
            
            # Generate answer using LLM
            answer = self.llm_service.generate_answer(
                query=query,
                context=context,
                max_tokens=500
            )
            
            state["answer"] = answer
            
            ProcessingStep.log_step(
                state, 
                "ANSWER_RESULT", 
                f"Generated answer ({len(answer)} chars): {answer[:100]}..."
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Answer node failed: {e}")
            raise WorkflowError(f"Answer node execution failed: {e}")
    
    @trace_node("reflect")
    def reflect_node(self, state: AgentState) -> AgentState:
        """
        Reflect node: Evaluate answer relevance and quality.
        
        This node:
        1. Analyzes the generated answer for relevance
        2. Assesses the quality and completeness
        3. Provides confidence scores and suggestions
        """
        try:
            query = state["query"]
            answer = state.get("answer", "")
            retrieved_docs = state.get("retrieved_docs", [])
            
            ProcessingStep.log_step(state, "REFLECT", "Evaluating answer quality and relevance")
            
            # Prepare context for reflection
            context = "\n".join(retrieved_docs) if retrieved_docs else "No context used"
            
            # Perform reflection using LLM
            reflection_result = self.llm_service.reflect_on_answer(
                query=query,
                answer=answer,
                context=context
            )
            
            # Update state with reflection results
            state["reflection"] = reflection_result.reasoning
            state["confidence_score"] = reflection_result.confidence_score
            
            ProcessingStep.log_step(
                state,
                "REFLECT_RESULT",
                f"Relevance: {reflection_result.is_relevant}, "
                f"Confidence: {reflection_result.confidence_score:.2f}, "
                f"Quality: {reflection_result.answer_quality}"
            )
            
            if reflection_result.suggestions:
                ProcessingStep.log_step(
                    state,
                    "SUGGESTIONS",
                    f"Improvements: {', '.join(reflection_result.suggestions)}"
                )
            
            # Calculate total processing time
            start_time = state.get("start_time", time.time())
            processing_time = time.time() - start_time
            
            ProcessingStep.log_step(
                state,
                "COMPLETE",
                f"Total processing time: {processing_time:.2f} seconds"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Reflect node failed: {e}")
            raise WorkflowError(f"Reflect node execution failed: {e}")
    
    def _analyze_query_for_retrieval(self, query: str) -> str:
        """
        Analyze query to determine if retrieval is needed.
        
        This is a simple heuristic-based approach. In a production system,
        this could be replaced with a more sophisticated classifier.
        """
        query_lower = query.lower().strip()
        
        # Queries that typically don't need retrieval
        no_retrieval_patterns = [
            "hello", "hi", "thanks", "thank you", "goodbye", "bye",
            "what is your name", "who are you", "how are you"
        ]
        
        # Check for greeting/conversational patterns
        for pattern in no_retrieval_patterns:
            if pattern in query_lower:
                return "no_retrieval"
        
        # Very short queries might not need retrieval
        if len(query.split()) <= 2 and not any(word in query_lower for word in ["what", "how", "why", "when", "where"]):
            return "no_retrieval"
        
        # Question words typically indicate need for retrieval
        question_words = ["what", "how", "why", "when", "where", "who", "which", "explain", "describe", "tell me"]
        
        if any(word in query_lower for word in question_words):
            return "retrieve"
        
        # Default to retrieval for informational queries
        return "retrieve"

# Utility functions for testing individual nodes
def test_workflow_nodes():
    """Test individual workflow nodes."""
    print("Testing Workflow Nodes...")
    
    # Initialize nodes
    nodes = WorkflowNodes()
    
    # Test state
    test_state: AgentState = {
        "query": "What are the benefits of renewable energy?",
        "plan_decision": "",
        "retrieved_docs": [],
        "answer": "",
        "reflection": "",
        "confidence_score": 0.0,
        "processing_steps": [],
        "start_time": time.time()
    }
    
    print(f"Initial query: {test_state['query']}")
    
    # Test plan node
    print("\n--- Testing Plan Node ---")
    test_state = nodes.plan_node(test_state)
    print(f"Plan decision: {test_state['plan_decision']}")
    
    # Test retrieve node
    print("\n--- Testing Retrieve Node ---")
    test_state = nodes.retrieve_node(test_state)
    print(f"Retrieved {len(test_state['retrieved_docs'])} documents")
    
    # Test answer node
    print("\n--- Testing Answer Node ---")
    test_state = nodes.answer_node(test_state)
    print(f"Generated answer: {test_state['answer'][:100]}...")
    
    # Test reflect node
    print("\n--- Testing Reflect Node ---")
    test_state = nodes.reflect_node(test_state)
    print(f"Reflection: {test_state['reflection'][:100]}...")
    print(f"Confidence: {test_state['confidence_score']:.2f}")
    
    # Show processing steps
    print("\n--- Processing Steps ---")
    for step in test_state.get('processing_steps', []):
        print(f"  {step}")
    
    print("Workflow nodes test completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_workflow_nodes()