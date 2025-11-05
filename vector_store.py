"""Vector database integration using ChromaDB."""

import os
import logging
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from models import Document, RetrievalResult, VectorStoreError
from config import Config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using sentence transformers."""
    
    def __init__(self, model_name: str = None):
        """Initialize the embedding service."""
        self.model_name = model_name or Config.EMBEDDING_MODEL
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise VectorStoreError(f"Failed to initialize embedding model: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise VectorStoreError(f"Embedding generation failed: {e}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise VectorStoreError(f"Batch embedding generation failed: {e}")

class VectorStore:
    """ChromaDB-based vector store for document retrieval."""
    
    def __init__(self, collection_name: str = None, persist_dir: str = None):
        """Initialize the vector store."""
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.persist_dir = persist_dir or Config.CHROMA_PERSIST_DIR
        self.embedding_service = EmbeddingService()
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Initialized vector store with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise VectorStoreError(f"ChromaDB initialization failed: {e}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents provided for addition")
            return
        
        try:
            # Prepare data for ChromaDB
            texts = [doc.content for doc in documents]
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                # Create unique ID
                doc_id = f"doc_{i}_{hash(doc.content[:100])}"
                ids.append(doc_id)
                
                # Prepare metadata
                metadata = doc.metadata.copy()
                metadata['source'] = doc.source
                metadata['content_length'] = len(doc.content)
                metadatas.append(metadata)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self.embedding_service.embed_texts(texts)
            
            # Add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise VectorStoreError(f"Document addition failed: {e}")
    
    def similarity_search(self, query: str, k: int = None) -> RetrievalResult:
        """Perform similarity search for relevant documents."""
        k = k or Config.TOP_K_DOCUMENTS
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_text(query)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            documents = []
            scores = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (cosine distance -> similarity)
                    similarity_score = 1 - distance
                    
                    # Create document object
                    document = Document(
                        content=doc_text,
                        metadata=metadata,
                        source=metadata.get('source', 'unknown')
                    )
                    
                    documents.append(document)
                    scores.append(similarity_score)
            
            # Get total document count
            total_docs = self.collection.count()
            
            logger.info(f"Retrieved {len(documents)} documents for query: '{query[:50]}...'")
            
            return RetrievalResult(
                documents=documents,
                scores=scores,
                query=query,
                total_docs_searched=total_docs
            )
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise VectorStoreError(f"Similarity search failed: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_dir': self.persist_dir,
                'embedding_model': self.embedding_service.model_name
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {'error': str(e)}
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise VectorStoreError(f"Collection clearing failed: {e}")
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete specific documents by their IDs."""
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise VectorStoreError(f"Document deletion failed: {e}")

# Utility functions for testing
def test_vector_store():
    """Test the vector store functionality."""
    print("Testing Vector Store...")
    
    # Create test documents
    test_docs = [
        Document(
            content="Solar energy is a renewable energy source that harnesses sunlight.",
            metadata={'topic': 'renewable_energy', 'type': 'definition'},
            source="test_solar.txt"
        ),
        Document(
            content="Wind turbines convert kinetic energy from wind into electricity.",
            metadata={'topic': 'renewable_energy', 'type': 'definition'},
            source="test_wind.txt"
        ),
        Document(
            content="Machine learning algorithms can learn patterns from data.",
            metadata={'topic': 'artificial_intelligence', 'type': 'definition'},
            source="test_ml.txt"
        )
    ]
    
    # Initialize vector store
    vs = VectorStore(collection_name="test_collection")
    
    # Clear any existing data
    vs.clear_collection()
    
    # Add documents
    vs.add_documents(test_docs)
    
    # Test search
    result = vs.similarity_search("What is solar power?", k=2)
    
    print(f"Query: 'What is solar power?'")
    print(f"Retrieved {len(result.documents)} documents:")
    for i, (doc, score) in enumerate(zip(result.documents, result.scores)):
        print(f"  {i+1}. Score: {score:.3f}")
        print(f"     Content: {doc.content[:100]}...")
        print(f"     Source: {doc.source}")
    
    # Get collection info
    info = vs.get_collection_info()
    print(f"\nCollection Info: {info}")
    
    print("Vector store test completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_vector_store()