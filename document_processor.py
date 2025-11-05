"""Document processing pipeline for loading and chunking knowledge base files."""

import os
import logging
from typing import List, Dict, Any, Optional
import re
from pathlib import Path

try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("pypdf not available. PDF processing will be disabled.")

from models import Document, DocumentProcessingError
from config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles loading, cleaning, and chunking of documents."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """Initialize the document processor."""
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        logger.info(f"Initialized DocumentProcessor with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory."""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise DocumentProcessingError(f"Directory not found: {directory_path}")
        
        documents = []
        supported_extensions = ['.txt', '.md']
        if PDF_AVAILABLE:
            supported_extensions.append('.pdf')
        
        logger.info(f"Loading documents from: {directory_path}")
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    file_documents = self.load_document(str(file_path))
                    documents.extend(file_documents)
                    logger.info(f"Loaded {len(file_documents)} chunks from {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    continue
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document and return chunked documents."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        # Determine file type and load content
        if file_path.suffix.lower() == '.pdf':
            content = self._load_pdf(file_path)
        else:
            content = self._load_text_file(file_path)
        
        # Clean the content
        cleaned_content = self._clean_text(content)
        
        # Chunk the content
        chunks = self._chunk_text(cleaned_content)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = {
                'source_file': str(file_path),
                'chunk_index': i,
                'total_chunks': len(chunks),
                'file_type': file_path.suffix.lower(),
                'original_length': len(content)
            }
            
            document = Document(
                content=chunk,
                metadata=metadata,
                source=file_path.name
            )
            documents.append(document)
        
        return documents
    
    def _load_text_file(self, file_path: Path) -> str:
        """Load content from a text file."""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                logger.debug(f"Successfully loaded {file_path} with encoding {encoding}")
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise DocumentProcessingError(f"Failed to read {file_path}: {e}")
        
        raise DocumentProcessingError(f"Could not decode {file_path} with any supported encoding")
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load content from a PDF file."""
        if not PDF_AVAILABLE:
            raise DocumentProcessingError("PDF processing not available. Install pypdf package.")
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                content = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1} of {file_path}: {e}")
                        continue
                
                return content
                
        except Exception as e:
            raise DocumentProcessingError(f"Failed to read PDF {file_path}: {e}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence or paragraph
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(start + self.chunk_size - 200, start)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                
                if sentence_end > start:
                    end = sentence_end
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            if end >= len(text):
                break
            
            start = end - self.chunk_overlap
            
            # Ensure we make progress (don't go backwards)
            if start < 0:
                start = 0
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within the given range."""
        # Look for sentence endings (., !, ?) followed by whitespace or end of text
        sentence_pattern = r'[.!?]\s+'
        
        # Search backwards from end to start
        search_text = text[start:end]
        matches = list(re.finditer(sentence_pattern, search_text))
        
        if matches:
            # Return position after the last sentence ending
            last_match = matches[-1]
            return start + last_match.end()
        
        # If no sentence boundary found, look for paragraph breaks
        paragraph_pattern = r'\n\s*\n'
        matches = list(re.finditer(paragraph_pattern, search_text))
        
        if matches:
            last_match = matches[-1]
            return start + last_match.start()
        
        # If no good boundary found, return the original end
        return end
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about the processed documents."""
        if not documents:
            return {'total_documents': 0}
        
        total_docs = len(documents)
        total_chars = sum(len(doc.content) for doc in documents)
        avg_chunk_size = total_chars / total_docs if total_docs > 0 else 0
        
        # Group by source file
        source_files = {}
        for doc in documents:
            source = doc.metadata.get('source_file', 'unknown')
            if source not in source_files:
                source_files[source] = 0
            source_files[source] += 1
        
        # File type distribution
        file_types = {}
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            if file_type not in file_types:
                file_types[file_type] = 0
            file_types[file_type] += 1
        
        return {
            'total_documents': total_docs,
            'total_characters': total_chars,
            'average_chunk_size': round(avg_chunk_size, 2),
            'source_files': source_files,
            'file_types': file_types,
            'chunk_size_setting': self.chunk_size,
            'chunk_overlap_setting': self.chunk_overlap
        }

def create_sample_documents():
    """Create sample documents for testing if knowledge base is empty."""
    kb_dir = Path(Config.KNOWLEDGE_BASE_DIR)
    kb_dir.mkdir(exist_ok=True)
    
    # Check if directory is empty
    existing_files = list(kb_dir.glob('*.txt'))
    if existing_files:
        logger.info(f"Knowledge base already contains {len(existing_files)} files")
        return
    
    logger.info("Creating sample documents for testing...")
    
    # Sample document about Python programming
    python_content = """
Python Programming Basics

Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support
- Versatile applications (web development, data science, AI, automation)

Basic Syntax:
Python uses indentation to define code blocks, making it highly readable. Variables don't need explicit declaration, and the language supports multiple programming paradigms.

Popular Libraries:
- NumPy: Numerical computing
- Pandas: Data manipulation and analysis
- Django/Flask: Web frameworks
- TensorFlow/PyTorch: Machine learning
- Matplotlib: Data visualization

Applications:
Python is widely used in web development, data science, artificial intelligence, scientific computing, automation, and more.
"""
    
    # Sample document about data science
    data_science_content = """
Data Science Overview

Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.

Key Components:
1. Statistics and Mathematics
2. Programming (Python, R, SQL)
3. Domain Expertise
4. Data Visualization
5. Machine Learning

Data Science Process:
1. Problem Definition
2. Data Collection
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis
5. Model Building and Training
6. Model Evaluation
7. Deployment and Monitoring

Common Tools:
- Programming Languages: Python, R, SQL
- Visualization: Matplotlib, Seaborn, Plotly, Tableau
- Machine Learning: Scikit-learn, TensorFlow, PyTorch
- Big Data: Spark, Hadoop
- Databases: PostgreSQL, MongoDB, Cassandra

Career Opportunities:
Data scientists work in various industries including technology, finance, healthcare, retail, and government, solving complex problems through data analysis.
"""
    
    # Write sample files
    with open(kb_dir / 'python_programming.txt', 'w', encoding='utf-8') as f:
        f.write(python_content)
    
    with open(kb_dir / 'data_science.txt', 'w', encoding='utf-8') as f:
        f.write(data_science_content)
    
    logger.info("Created sample documents: python_programming.txt, data_science.txt")

# Utility functions for testing
def test_document_processor():
    """Test the document processor functionality."""
    print("Testing Document Processor...")
    
    # Create sample documents if needed
    create_sample_documents()
    
    # Initialize processor
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    
    # Load documents from knowledge base
    kb_dir = Config.KNOWLEDGE_BASE_DIR
    documents = processor.load_documents_from_directory(kb_dir)
    
    print(f"Loaded {len(documents)} document chunks")
    
    # Show first few documents
    for i, doc in enumerate(documents[:3]):
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.source}")
        print(f"  Content length: {len(doc.content)}")
        print(f"  Content preview: {doc.content[:100]}...")
        print(f"  Metadata: {doc.metadata}")
    
    # Get statistics
    stats = processor.get_document_stats(documents)
    print(f"\nDocument Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("Document processor test completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_document_processor()