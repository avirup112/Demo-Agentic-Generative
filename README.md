# RAG Q&A Agent

A Retrieval-Augmented Generation (RAG) Question-Answering system built with LangGraph that can answer questions from a local knowledge base using vector similarity search and large language models.

## 🚀 Features

- **LangGraph Workflow**: Four-node pipeline (Plan → Retrieve → Answer → Reflect)
- **Vector Database**: ChromaDB for efficient document storage and retrieval
- **Groq Integration**: Fast LLM inference with Llama3/Mixtral models
- **Document Processing**: Support for TXT, MD, and PDF files
- **Interactive & Batch Modes**: CLI interface for single queries or batch processing
- **Comprehensive Evaluation**: ROUGE, BERTScore, LLM-as-Judge, and RAGAs metrics
- **Streamlit Web UI**: Interactive web interface with real-time evaluation
- **LangSmith Tracing**: Advanced tracing and monitoring (optional)
- **Comprehensive Logging**: Detailed step-by-step processing logs
- **Mock Fallback**: Works without API key for testing

## 📋 Requirements

- Python 3.8+
- Groq API key (optional - system works with mock responses)
- Dependencies listed in `requirements.txt`

> ⚠️ **Model Update**: Some Groq models have been deprecated. See [MODEL_MIGRATION.md](MODEL_MIGRATION.md) for current models.

## 🛠️ Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Configure API keys in the web interface:**
   - Get your Groq API key from [console.groq.com](https://console.groq.com/)
   - Enter it in the sidebar under "API Keys"
   - Optionally add LangSmith key for advanced tracing
   - Start asking questions!

## 🎯 Quick Start

### One-Command Setup
```bash
pip install -r requirements.txt && streamlit run streamlit_app.py
```

### Dynamic Configuration
No need for .env files! Configure everything in the web interface:
- **API Keys**: Enter Groq API key directly in the sidebar
- **Model Selection**: Choose from available Groq models
- **Optional Tracing**: Add LangSmith credentials for monitoring
- **Real-time Updates**: Changes apply immediately

The web interface provides:
- Interactive Q&A with the knowledge base
- Real-time evaluation metrics
- Query history and export
- Dataset evaluation tools
- Dynamic system configuration

### Features Available in Web Interface
- **Interactive Q&A**: Ask questions in real-time
- **Batch Processing**: Upload question files for bulk processing
- **Evaluation Dashboard**: View comprehensive metrics
- **Knowledge Base Management**: Setup and configure documents
- **Export Results**: Download query history and evaluations

## 📁 Project Structure

```
rag-qa-agent/
├── streamlit_app.py       # Web interface (main entry point)
├── rag_agent.py           # Core RAG agent with LangGraph
├── workflow_nodes.py      # LangGraph workflow nodes
├── vector_store.py        # ChromaDB integration
├── llm_service.py         # Groq LLM service
├── document_processor.py  # Document loading and chunking
├── evaluation.py          # Evaluation framework
├── tracing.py             # LangSmith tracing
├── models.py              # Data models and types
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── knowledge_base/        # Directory for knowledge base files
│   ├── renewable_energy.txt
│   ├── artificial_intelligence.txt
│   └── climate_change.txt
└── chroma_db/            # ChromaDB storage (created automatically)
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# API Keys
GROQ_API_KEY=your_groq_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional

# Model Configuration
LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Retrieval Settings
TOP_K_DOCUMENTS=3
SIMILARITY_THRESHOLD=0.7

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Tracing and Evaluation
ENABLE_TRACING=false
ENABLE_EVALUATION=true
LANGSMITH_PROJECT=rag-qa-agent
```

### Adding Documents

1. Place your documents (TXT, MD, PDF) in the `knowledge_base/` directory
2. Run setup to reindex: `python main.py --setup`

## 🏗️ Architecture

### LangGraph Workflow

The agent uses a four-node LangGraph workflow:

1. **Plan Node**: Analyzes the query and decides if retrieval is needed
2. **Retrieve Node**: Performs vector similarity search in ChromaDB
3. **Answer Node**: Generates response using LLM and retrieved context
4. **Reflect Node**: Evaluates answer quality and relevance

### Components

- **Vector Store**: ChromaDB with sentence-transformers embeddings
- **LLM Service**: Groq integration with mock fallback
- **Document Processor**: Text chunking and preprocessing
- **Workflow Orchestration**: LangGraph state management

## 📊 Usage Examples

### Python API

```python
from rag_agent import RAGAgent

# Initialize agent
agent = RAGAgent(initialize_kb=True)

# Ask a question
result = agent.query("What are the benefits of renewable energy?")

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Processing time: {result.processing_time}s")
```

### Command Line

```bash
# Interactive mode
python main.py

# Batch processing
python main.py --batch questions.txt --output results.json

# Setup with custom log level
python main.py --setup --log-level DEBUG
```

## 🧪 Testing

### Test Individual Components

```bash
# Test vector store
python -c "from vector_store import test_vector_store; test_vector_store()"

# Test LLM service  
python -c "from llm_service import test_llm_service; test_llm_service()"

# Test document processor
python -c "from document_processor import test_document_processor; test_document_processor()"

# Test complete system
streamlit run streamlit_app.py
```

## 🔍 Monitoring & Debugging

### Logging

The system provides comprehensive logging at multiple levels:

- **INFO**: General operation status
- **DEBUG**: Detailed processing steps
- **ERROR**: Error conditions and stack traces

Logs are written to both console and `rag_agent.log` file.

### Processing Steps

Each query shows step-by-step processing:

```
🔄 [1699123456.78] PLAN: Analyzing query: 'What are the benefits of renewable energy?'
🔄 [1699123456.82] PLAN_RESULT: Decision: retrieve
🔄 [1699123456.85] RETRIEVE: Searching knowledge base for: 'What are the benefits...'
🔄 [1699123457.12] RETRIEVE_RESULT: Retrieved 3 relevant documents
🔄 [1699123457.15] ANSWER: Generating answer for: 'What are the benefits...'
🔄 [1699123458.45] ANSWER_RESULT: Generated answer (245 chars): Based on the provided context...
🔄 [1699123458.48] REFLECT: Evaluating answer quality and relevance
🔄 [1699123459.23] REFLECT_RESULT: Relevance: True, Confidence: 0.85, Quality: good
```

## 🚨 Troubleshooting

### Common Issues

1. **No Groq API Key**: System works with mock responses for testing
2. **Empty Knowledge Base**: Run `python main.py --setup` to create sample documents
3. **ChromaDB Issues**: Delete `chroma_db/` directory and reinitialize
4. **Memory Issues**: Reduce `CHUNK_SIZE` and `TOP_K_DOCUMENTS` in config

### Error Messages

- **"Knowledge base initialization failed"**: Check document directory permissions
- **"LLM service initialization failed"**: Verify API key or use mock mode
- **"Vector store error"**: Clear and reinitialize ChromaDB

## 📊 Evaluation Framework

The system includes comprehensive evaluation capabilities:

### Metrics Available
- **ROUGE Scores**: Text overlap similarity (ROUGE-1, ROUGE-2, ROUGE-L)
- **BERTScore**: Semantic similarity using BERT embeddings
- **LLM-as-Judge**: Quality assessment using the same LLM
- **RAGAs**: Specialized RAG evaluation metrics (optional)

### Usage
```bash
# Run evaluation on sample dataset
python main.py --evaluate

# Use web interface for interactive evaluation
streamlit run streamlit_app.py
```

## 🔍 Tracing and Monitoring

Optional LangSmith integration for advanced tracing:

1. Get API key from [LangSmith](https://smith.langchain.com/)
2. Set `LANGSMITH_API_KEY` in `.env`
3. Enable tracing: `ENABLE_TRACING=true`
4. View traces in LangSmith dashboard

## 🌐 Web Interface

Interactive Streamlit interface with:
- Real-time query processing
- Evaluation metrics display
- Query history management
- Dataset evaluation tools
- Visualization of results

## 🔮 Future Enhancements

- Advanced RAG techniques (hybrid search, re-ranking)
- Multi-modal document support
- Real-time evaluation dashboards
- Custom evaluation metrics
- Multi-language support

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 🤝 Contributing

This is a demonstration project. For production use, consider:

- Adding comprehensive error handling
- Implementing proper authentication
- Adding rate limiting and caching
- Enhancing security measures
- Adding comprehensive test suite

## 📞 Support

For questions or issues:

1. Check the troubleshooting section
2. Review the logs in `rag_agent.log`
3. Test individual components using their test functions
4. Verify configuration in `.env` file

---

**Note**: This RAG Q&A Agent demonstrates modern AI workflows using LangGraph and Groq. The system is production-ready with comprehensive error handling and logging.