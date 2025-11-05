"""LLM service integration for answer generation and reflection."""

import logging
from typing import Dict, Any, List, Optional
from groq import Groq

from models import ReflectionResult, LLMServiceError
from config import Config

logger = logging.getLogger(__name__)

class LLMService:
    """Service for LLM-based answer generation and reflection."""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """Initialize the LLM service."""
        self.model_name = model_name or Config.LLM_MODEL
        self.api_key = api_key or Config.GROQ_API_KEY
        
        if not self.api_key:
            logger.warning("No Groq API key provided. Using mock responses.")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Initialized LLM service with Groq model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                raise LLMServiceError(f"LLM service initialization failed: {e}")
    
    def generate_answer(self, query: str, context: str, max_tokens: int = 500) -> str:
        """Generate an answer using the LLM with retrieved context."""
        
        # Create the prompt
        prompt = self._create_answer_prompt(query, context)
        
        if not self.client:
            # Return mock response if no API key
            return self._generate_mock_answer(query, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that provides accurate, informative answers based on the given context. If the context doesn't contain enough information to answer the question, say so clearly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer for query: '{query[:50]}...'")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise LLMServiceError(f"Answer generation failed: {e}")
    
    def reflect_on_answer(self, query: str, answer: str, context: str) -> ReflectionResult:
        """Evaluate the quality and relevance of the generated answer."""
        
        # Create reflection prompt
        prompt = self._create_reflection_prompt(query, answer, context)
        
        if not self.client:
            # Return mock reflection if no API key
            return self._generate_mock_reflection(query, answer, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an AI evaluator that assesses the quality and relevance of answers. Provide structured feedback in the requested format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            reflection_text = response.choices[0].message.content.strip()
            reflection_result = self._parse_reflection_response(reflection_text)
            
            logger.info(f"Generated reflection with confidence: {reflection_result.confidence_score:.2f}")
            return reflection_result
            
        except Exception as e:
            logger.error(f"Failed to generate reflection: {e}")
            # Return a default reflection on error
            return ReflectionResult(
                is_relevant=True,
                confidence_score=0.5,
                reasoning=f"Error during reflection: {str(e)}",
                suggestions=["Manual review recommended due to reflection error"],
                answer_quality="unknown"
            )
    
    def _create_answer_prompt(self, query: str, context: str) -> str:
        """Create a prompt for answer generation."""
        return f"""Based on the following context, please answer the user's question. If the context doesn't contain enough information to fully answer the question, please say so and provide what information you can.

Context:
{context}

Question: {query}

Please provide a clear, accurate, and helpful answer based on the context provided."""
    
    def _create_reflection_prompt(self, query: str, answer: str, context: str) -> str:
        """Create a prompt for answer reflection."""
        return f"""Please evaluate the following answer for relevance and quality. Provide your assessment in this exact format:

RELEVANCE: [Yes/No]
CONFIDENCE: [0.0-1.0]
QUALITY: [excellent/good/fair/poor]
REASONING: [Brief explanation]
SUGGESTIONS: [Comma-separated list of improvements, or "None" if no suggestions]

Original Question: {query}

Context Used:
{context}

Generated Answer:
{answer}

Please evaluate whether the answer is relevant to the question, how confident you are in its accuracy, and provide suggestions for improvement if needed."""
    
    def _parse_reflection_response(self, reflection_text: str) -> ReflectionResult:
        """Parse the structured reflection response."""
        try:
            lines = reflection_text.strip().split('\n')
            
            # Initialize default values
            is_relevant = True
            confidence_score = 0.7
            answer_quality = "good"
            reasoning = "Reflection parsing incomplete"
            suggestions = []
            
            # Parse each line
            for line in lines:
                line = line.strip()
                if line.startswith('RELEVANCE:'):
                    is_relevant = 'yes' in line.lower()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence_score = float(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        confidence_score = 0.7
                elif line.startswith('QUALITY:'):
                    quality_text = line.split(':')[1].strip().lower()
                    if quality_text in ['excellent', 'good', 'fair', 'poor']:
                        answer_quality = quality_text
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
                elif line.startswith('SUGGESTIONS:'):
                    suggestions_text = line.split(':', 1)[1].strip()
                    if suggestions_text.lower() != 'none':
                        suggestions = [s.strip() for s in suggestions_text.split(',')]
            
            return ReflectionResult(
                is_relevant=is_relevant,
                confidence_score=confidence_score,
                reasoning=reasoning,
                suggestions=suggestions,
                answer_quality=answer_quality
            )
            
        except Exception as e:
            logger.error(f"Failed to parse reflection response: {e}")
            return ReflectionResult(
                is_relevant=True,
                confidence_score=0.5,
                reasoning=f"Parsing error: {str(e)}",
                suggestions=["Manual review recommended"],
                answer_quality="unknown"
            )
    
    def _generate_mock_answer(self, query: str, context: str) -> str:
        """Generate a mock answer when no API key is available."""
        logger.info("Generating mock answer (no Groq API key provided)")
        
        # Simple keyword-based mock responses
        query_lower = query.lower()
        context_lower = context.lower()
        
        if 'renewable energy' in query_lower or 'solar' in query_lower or 'wind' in query_lower:
            return "Based on the provided context, renewable energy sources like solar and wind power offer significant environmental and economic benefits. They help reduce greenhouse gas emissions and provide sustainable alternatives to fossil fuels."
        
        elif 'artificial intelligence' in query_lower or 'ai' in query_lower or 'machine learning' in query_lower:
            return "According to the context, artificial intelligence encompasses various technologies including machine learning, which enables systems to learn from data and make intelligent decisions across multiple domains."
        
        elif 'climate change' in query_lower or 'global warming' in query_lower:
            return "The context indicates that climate change is primarily driven by human activities, particularly greenhouse gas emissions, and requires both mitigation and adaptation strategies to address its impacts."
        
        else:
            return f"Based on the available context, I can provide information related to your question about '{query}'. However, I'm currently operating in demo mode. For more detailed and accurate responses, please configure a Groq API key."
    
    def _generate_mock_reflection(self, query: str, answer: str, context: str) -> ReflectionResult:
        """Generate a mock reflection when no API key is available."""
        logger.info("Generating mock reflection (no Groq API key provided)")
        
        # Simple heuristic-based reflection
        answer_length = len(answer.split())
        context_length = len(context.split())
        
        # Basic relevance check
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words.intersection(answer_words))
        
        is_relevant = overlap > 0
        confidence_score = min(0.9, 0.5 + (overlap * 0.1) + (answer_length * 0.01))
        
        if answer_length < 10:
            answer_quality = "poor"
            suggestions = ["Provide more detailed information", "Include specific examples"]
        elif answer_length < 30:
            answer_quality = "fair"
            suggestions = ["Add more context", "Include supporting details"]
        else:
            answer_quality = "good"
            suggestions = []
        
        return ReflectionResult(
            is_relevant=is_relevant,
            confidence_score=confidence_score,
            reasoning=f"Mock evaluation based on answer length ({answer_length} words) and keyword overlap",
            suggestions=suggestions,
            answer_quality=answer_quality
        )

# Utility functions for testing
def test_llm_service():
    """Test the LLM service functionality."""
    print("Testing LLM Service...")
    
    # Initialize service
    llm_service = LLMService()
    
    # Test data
    query = "What are the benefits of renewable energy?"
    context = """
    Renewable energy sources offer numerous benefits including:
    - Reduced greenhouse gas emissions
    - Energy independence and security
    - Job creation in green industries
    - Stable long-term energy costs
    - Minimal environmental impact
    """
    
    # Test answer generation
    print(f"Query: {query}")
    print(f"Context: {context[:100]}...")
    
    answer = llm_service.generate_answer(query, context)
    print(f"Generated Answer: {answer}")
    
    # Test reflection
    reflection = llm_service.reflect_on_answer(query, answer, context)
    print(f"\nReflection Results:")
    print(f"  Relevant: {reflection.is_relevant}")
    print(f"  Confidence: {reflection.confidence_score:.2f}")
    print(f"  Quality: {reflection.answer_quality}")
    print(f"  Reasoning: {reflection.reasoning}")
    print(f"  Suggestions: {reflection.suggestions}")
    
    print("LLM service test completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_llm_service()