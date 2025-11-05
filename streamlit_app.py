"""Streamlit web interface for the RAG Q&A Agent with evaluation features."""

import streamlit as st
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# Handle optional imports gracefully
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Install with: pip install plotly")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    st.warning("Pandas not available. Install with: pip install pandas")

# Import our modules with error handling
try:
    from rag_agent import RAGAgent
    from evaluation import RAGEvaluator, create_sample_evaluation_dataset
    from models import QueryResult
    from config import Config
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"Failed to import required modules: {e}")
    st.info("Please ensure all dependencies are installed: pip install -r requirements.txt")

# Page configuration
st.set_page_config(
    page_title="RAG Q&A Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Universal color scheme that works on both light and dark themes
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .answer-box {
        background: linear-gradient(135deg, #2C3E50, #34495E);
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #FF6B35;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .answer-box p, .answer-box div, .answer-box span {
        color: #FFFFFF !important;
        font-weight: 500;
        line-height: 1.6;
        text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
    }
    
    .evaluation-box {
        background: linear-gradient(135deg, #27AE60, #2ECC71);
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #FFFFFF;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .evaluation-box p, .evaluation-box div, .evaluation-box span {
        color: #FFFFFF !important;
        font-weight: 500;
        text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
    }
    
    .processing-step {
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #FFFFFF;
        background: linear-gradient(135deg, #8E44AD, #9B59B6);
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        border-left: 4px solid #F39C12;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #3498DB, #2980B9);
        color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 10px;
        border: 2px solid #FFFFFF;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .metric-card p, .metric-card div, .metric-card span {
        color: #FFFFFF !important;
        font-weight: 500;
    }
    
    /* Override Streamlit's default text colors */
    .stMarkdown, .stText, .element-container {
        color: inherit !important;
    }
    
    /* Remove hover effects */
    .answer-box:hover, .evaluation-box:hover, .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
    }
    
    /* Ensure all markdown text is visible */
    div[data-testid="stMarkdownContainer"] {
        color: inherit !important;
    }
    
    div[data-testid="stMarkdownContainer"] p {
        color: inherit !important;
        font-weight: 500 !important;
    }
    
    /* Button styling for better visibility */
    .stButton > button {
        background: linear-gradient(135deg, #E74C3C, #C0392B);
        color: #FFFFFF;
        border: 2px solid #FFFFFF;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0 3px 6px rgba(0,0,0,0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #C0392B, #A93226);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2C3E50, #34495E);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #34495E;
        color: #FFFFFF;
        border: 2px solid #FF6B35;
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input::placeholder, 
    .stTextArea > div > div > textarea::placeholder {
        color: #BDC3C7;
    }
</style>
""", unsafe_allow_html=True)

def initialize_agent_dynamic():
    """Initialize the RAG agent with dynamic configuration."""
    if not MODULES_AVAILABLE:
        return None, "Required modules not available"
    
    try:
        # Check if we have API key in environment (set by sidebar)
        import os
        if os.getenv("GROQ_API_KEY"):
            with st.spinner("Initializing RAG Agent..."):
                agent = RAGAgent(initialize_kb=True)
            return agent, None
        else:
            # Create agent without API key (mock mode)
            with st.spinner("Initializing RAG Agent (Mock Mode)..."):
                agent = RAGAgent(initialize_kb=True)
            return agent, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def initialize_evaluator(_agent):
    """Initialize the evaluator with caching."""
    try:
        evaluator = RAGEvaluator(_agent)
        return evaluator, None
    except Exception as e:
        return None, str(e)

def display_query_result(result: QueryResult, show_details: bool = True):
    """Display query result in a formatted way."""
    
    # Answer section
    st.markdown("### 🎯 Answer")
    st.markdown(f'<div class="answer-box">{result.answer}</div>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Confidence", f"{result.confidence:.2f}")
    
    with col2:
        st.metric("Processing Time", f"{result.processing_time:.2f}s")
    
    with col3:
        st.metric("Retrieved Docs", len(result.retrieved_docs))
    
    with col4:
        quality_score = "High" if result.confidence > 0.8 else "Medium" if result.confidence > 0.5 else "Low"
        st.metric("Quality", quality_score)
    
    if show_details:
        # Expandable sections for detailed information
        with st.expander("📄 Retrieved Documents"):
            if result.retrieved_docs:
                for i, doc in enumerate(result.retrieved_docs, 1):
                    st.markdown(f"**Document {i}:**")
                    st.text(doc.content[:300] + "..." if len(doc.content) > 300 else doc.content)
                    if hasattr(doc, 'metadata') and doc.metadata:
                        st.json(doc.metadata)
                    st.markdown("---")
            else:
                st.info("No documents were retrieved for this query.")
        
        with st.expander("🤔 Reflection & Analysis"):
            if result.reflection:
                st.markdown(f"**Reflection:** {result.reflection}")
            else:
                st.info("No reflection available.")
        
        with st.expander("🔄 Processing Steps"):
            if result.processing_steps:
                for step in result.processing_steps:
                    st.markdown(f'<div class="processing-step">{step}</div>', unsafe_allow_html=True)
            else:
                st.info("No processing steps recorded.")

def display_evaluation_results(eval_results: Dict[str, Any]):
    """Display evaluation results with visualizations."""
    
    if not eval_results:
        st.warning("No evaluation results to display.")
        return
    
    st.markdown("### 📊 Evaluation Results")
    
    # Individual result
    if "rouge1" in eval_results:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="evaluation-box">', unsafe_allow_html=True)
            st.metric("ROUGE-1", f"{eval_results.get('rouge1', 0):.3f}")
            st.metric("ROUGE-2", f"{eval_results.get('rouge2', 0):.3f}")
            st.metric("ROUGE-L", f"{eval_results.get('rougeL', 0):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="evaluation-box">', unsafe_allow_html=True)
            st.metric("Relevance", f"{eval_results.get('relevance', 0):.1f}/5")
            st.metric("Accuracy", f"{eval_results.get('accuracy', 0):.1f}/5")
            st.metric("Overall", f"{eval_results.get('overall', 0):.1f}/5")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            if any(key.startswith('bert_') for key in eval_results.keys()):
                st.markdown('<div class="evaluation-box">', unsafe_allow_html=True)
                st.metric("BERT Precision", f"{eval_results.get('bert_precision', 0):.3f}")
                st.metric("BERT Recall", f"{eval_results.get('bert_recall', 0):.3f}")
                st.metric("BERT F1", f"{eval_results.get('bert_f1', 0):.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Dataset evaluation results
    elif "aggregate_metrics" in eval_results:
        aggregate = eval_results["aggregate_metrics"]
        individual = eval_results.get("individual_results", [])
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Dataset Size", eval_results.get("dataset_size", 0))
        with col2:
            st.metric("Avg Confidence", f"{aggregate.get('avg_confidence', 0):.3f}")
        with col3:
            st.metric("Avg ROUGE-1", f"{aggregate.get('avg_rouge1', 0):.3f}")
        with col4:
            st.metric("Avg Quality", f"{aggregate.get('avg_overall_quality', 0):.1f}/5")
        
        # Detailed metrics
        st.markdown("#### 📈 Detailed Metrics")
        
        metrics_data = {
            "Metric": [],
            "Value": []
        }
        
        for key, value in aggregate.items():
            if key.startswith('avg_'):
                clean_name = key.replace('avg_', '').replace('_', ' ').title()
                metrics_data["Metric"].append(clean_name)
                metrics_data["Value"].append(value)
        
        if PANDAS_AVAILABLE and PLOTLY_AVAILABLE:
            df_metrics = pd.DataFrame(metrics_data)
            
            # Create bar chart
            fig = px.bar(df_metrics, x="Metric", y="Value", 
                        title="Average Evaluation Metrics")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback display without charts
            for metric, value in zip(metrics_data["Metric"], metrics_data["Value"]):
                st.metric(metric, f"{value:.3f}")
        
        # Individual results distribution
        if individual:
            st.markdown("#### 📊 Score Distributions")
            
            # Create distribution plots
            col1, col2 = st.columns(2)
            
            if PLOTLY_AVAILABLE:
                with col1:
                    rouge1_scores = [r.get('rouge1', 0) for r in individual]
                    fig_rouge = px.histogram(x=rouge1_scores, nbins=10, 
                                           title="ROUGE-1 Score Distribution")
                    st.plotly_chart(fig_rouge, use_container_width=True)
                
                with col2:
                    overall_scores = [r.get('overall', 0) for r in individual]
                    fig_overall = px.histogram(x=overall_scores, nbins=5, 
                                             title="Overall Quality Distribution")
                    st.plotly_chart(fig_overall, use_container_width=True)
            else:
                st.info("Install plotly for visualization: pip install plotly")

def qa_interface():
    """Main Q&A interface tab."""
    
    # Initialize agent
    agent, error = initialize_agent_dynamic()
    
    if error:
        st.error(f"Failed to initialize agent: {error}")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys Configuration
        st.subheader("🔑 API Keys")
        
        # Groq API Key
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get your API key from https://console.groq.com/",
            placeholder="Enter your Groq API key..."
        )
        
        # Model Selection
        model_options = [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile", 
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        selected_model = st.selectbox(
            "LLM Model",
            model_options,
            help="Choose the Groq model to use"
        )
        
        # Optional: LangSmith for tracing
        with st.expander("🔍 Optional: LangSmith Tracing"):
            langsmith_key = st.text_input(
                "LangSmith API Key",
                type="password",
                help="Optional: For advanced tracing and monitoring",
                placeholder="Enter LangSmith API key (optional)..."
            )
            
            langsmith_project = st.text_input(
                "LangSmith Project Name",
                value="rag-qa-agent",
                help="Project name for LangSmith tracing"
            )
        
        # Apply configuration
        if groq_key:
            # Update configuration dynamically
            import os
            os.environ["GROQ_API_KEY"] = groq_key
            os.environ["LLM_MODEL"] = selected_model
            
            if langsmith_key:
                os.environ["LANGSMITH_API_KEY"] = langsmith_key
                os.environ["LANGSMITH_PROJECT"] = langsmith_project
                os.environ["ENABLE_TRACING"] = "true"
            
            st.success("✅ Configuration Applied!")
        else:
            st.warning("⚠️ No Groq API key provided. Using mock responses.")
        
        st.markdown("---")
        
        # Knowledge base info
        if groq_key:  # Only try to initialize if we have API key
            kb_info = agent.get_knowledge_base_info()
            if kb_info.get('status') == 'ready':
                collection_info = kb_info.get('collection_info', {})
                config = kb_info.get('config', {})
                
                st.success("Knowledge Base Ready")
                st.metric("Documents", collection_info.get('document_count', 0))
                
                with st.expander("📊 System Details"):
                    st.write(f"**Embedding Model:** {collection_info.get('embedding_model', 'unknown')}")
                    st.write(f"**LLM Model:** {selected_model}")
                    st.write(f"**Chunk Size:** {config.get('chunk_size', 'unknown')}")
                    st.write(f"**Top-K Retrieval:** {config.get('top_k_documents', 'unknown')}")
            else:
                st.error("Knowledge Base Error")
                st.write(kb_info.get('error', 'Unknown error'))
        else:
            st.info("💡 Add your Groq API key above to enable full functionality")
        
        st.markdown("---")
        
        # Settings
        st.header("🎛️ Query Settings")
        show_details = st.checkbox("Show detailed results", value=True)
        enable_evaluation = st.checkbox("Enable evaluation", value=False)
        auto_clear = st.checkbox("Auto-clear after query", value=False)
        
        if enable_evaluation:
            reference_answer = st.text_area(
                "Reference answer (for evaluation):",
                placeholder="Enter the expected answer for comparison..."
            )
        
        st.markdown("---")
        
        # Sample questions
        st.header("💡 Sample Questions")
        sample_questions = [
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "What causes climate change?",
            "Explain artificial intelligence applications",
            "Compare solar and wind energy"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}"):
                st.session_state.query_input = question
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        query = st.text_area(
            "🔍 Ask your question:",
            height=100,
            placeholder="Enter your question about the knowledge base...",
            key="query_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        ask_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.query_input = ""
        st.rerun()
    
    # Process query
    if ask_button and query.strip():
        with st.spinner("🔄 Processing your question..."):
            try:
                # Generate unique session ID
                session_id = f"streamlit_{int(time.time())}"
                
                # Process query
                result = agent.query(query.strip(), session_id)
                
                # Display results
                st.markdown("---")
                display_query_result(result, show_details)
                
                # Evaluation if enabled
                if enable_evaluation and reference_answer.strip():
                    st.markdown("---")
                    st.markdown("### 🔬 Evaluation")
                    
                    with st.spinner("Evaluating answer quality..."):
                        evaluator, eval_error = initialize_evaluator(agent)
                        
                        if evaluator and not eval_error:
                            eval_results = evaluator.evaluate_single_query(
                                query.strip(), reference_answer.strip()
                            )
                            display_evaluation_results(eval_results)
                        else:
                            st.error(f"Evaluation failed: {eval_error}")
                
                # Store in session state for history
                if 'query_history' not in st.session_state:
                    st.session_state.query_history = []
                
                history_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'query': query,
                    'result': result
                }
                
                if enable_evaluation and reference_answer.strip():
                    history_entry['evaluation'] = eval_results
                
                st.session_state.query_history.append(history_entry)
                
                # Auto-clear if enabled
                if auto_clear:
                    st.session_state.query_input = ""
                    time.sleep(1)
                    st.rerun()
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
    
    elif ask_button and not query.strip():
        st.warning("Please enter a question before clicking 'Ask Question'.")
    
    # Query history
    if 'query_history' in st.session_state and st.session_state.query_history:
        st.markdown("---")
        st.header("📚 Query History")
        
        # History controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"Total queries: {len(st.session_state.query_history)}")
        with col2:
            if st.button("📥 Export History"):
                export_history()
        with col3:
            if st.button("🗑️ Clear History"):
                st.session_state.query_history = []
                st.rerun()
        
        # Display history (most recent first)
        for i, entry in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            with st.expander(f"Query {len(st.session_state.query_history) - i + 1}: {entry['query'][:50]}..."):
                st.write(f"**Timestamp:** {entry['timestamp']}")
                st.write(f"**Query:** {entry['query']}")
                display_query_result(entry['result'], show_details=False)
                
                if 'evaluation' in entry:
                    st.markdown("**Evaluation:**")
                    display_evaluation_results(entry['evaluation'])

def evaluation_interface():
    """Evaluation interface tab."""
    
    st.header("🔬 Model Evaluation")
    
    # Initialize agent and evaluator
    agent, agent_error = initialize_agent_dynamic()
    if agent_error:
        st.error(f"Failed to initialize agent: {agent_error}")
        return
    
    evaluator, eval_error = initialize_evaluator(agent)
    if eval_error:
        st.error(f"Failed to initialize evaluator: {eval_error}")
        return
    
    # Evaluation options
    eval_mode = st.radio(
        "Choose evaluation mode:",
        ["Single Question", "Dataset Evaluation", "Create Sample Dataset"]
    )
    
    if eval_mode == "Single Question":
        st.subheader("📝 Single Question Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            question = st.text_area("Question:", placeholder="Enter your question...")
        
        with col2:
            reference = st.text_area("Reference Answer:", placeholder="Enter the expected answer...")
        
        if st.button("🔍 Evaluate") and question.strip() and reference.strip():
            with st.spinner("Evaluating..."):
                try:
                    eval_results = evaluator.evaluate_single_query(question.strip(), reference.strip())
                    
                    st.markdown("---")
                    st.subheader("📊 Evaluation Results")
                    
                    # Display the generated answer
                    st.markdown("**Generated Answer:**")
                    st.write(eval_results["generated_answer"])
                    
                    # Display evaluation metrics
                    display_evaluation_results(eval_results)
                    
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
    
    elif eval_mode == "Dataset Evaluation":
        st.subheader("📊 Dataset Evaluation")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload evaluation dataset (JSON format):",
            type=['json'],
            help="Upload a JSON file with questions and reference answers"
        )
        
        # Use sample dataset option
        use_sample = st.checkbox("Use sample dataset", value=True)
        
        if st.button("🚀 Run Evaluation"):
            with st.spinner("Running evaluation on dataset..."):
                try:
                    if uploaded_file:
                        # Save uploaded file temporarily
                        dataset_path = "temp_dataset.json"
                        with open(dataset_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    elif use_sample:
                        # Create sample dataset
                        create_sample_evaluation_dataset()
                        dataset_path = "evaluation_dataset.json"
                    else:
                        st.error("Please upload a dataset or use the sample dataset.")
                        return
                    
                    # Run evaluation
                    eval_results = evaluator.evaluate_dataset(dataset_path)
                    
                    if eval_results:
                        st.markdown("---")
                        st.subheader("📈 Dataset Evaluation Results")
                        
                        # Display results
                        display_evaluation_results(eval_results)
                        
                        # Generate and display report
                        report = evaluator.generate_evaluation_report(eval_results)
                        
                        with st.expander("📄 Detailed Report"):
                            st.markdown(report)
                        
                        # Save results
                        output_path = f"evaluation_results_{int(time.time())}.json"
                        evaluator.save_evaluation_results(eval_results, output_path)
                        
                        st.success(f"Results saved to: {output_path}")
                        
                        # Download button
                        with open(output_path, "r") as f:
                            st.download_button(
                                label="📥 Download Results",
                                data=f.read(),
                                file_name=output_path,
                                mime="application/json"
                            )
                    
                except Exception as e:
                    st.error(f"Dataset evaluation failed: {e}")
    
    elif eval_mode == "Create Sample Dataset":
        st.subheader("📝 Create Sample Dataset")
        
        st.write("This will create a sample evaluation dataset with predefined questions and answers.")
        
        if st.button("🔧 Create Sample Dataset"):
            try:
                create_sample_evaluation_dataset()
                st.success("✅ Sample dataset created: evaluation_dataset.json")
                
                # Show preview
                with open("evaluation_dataset.json", "r") as f:
                    sample_data = json.load(f)
                
                st.subheader("📋 Dataset Preview")
                for i, item in enumerate(sample_data[:3], 1):
                    with st.expander(f"Sample {i}: {item['question'][:50]}..."):
                        st.write(f"**Question:** {item['question']}")
                        st.write(f"**Reference Answer:** {item['reference_answer']}")
                
                # Download button
                with open("evaluation_dataset.json", "r") as f:
                    st.download_button(
                        label="📥 Download Sample Dataset",
                        data=f.read(),
                        file_name="evaluation_dataset.json",
                        mime="application/json"
                    )
                
            except Exception as e:
                st.error(f"Failed to create sample dataset: {e}")

def about_page():
    """About page with system information."""
    st.header("ℹ️ About RAG Q&A Agent")
    
    st.markdown("""
    This is a Retrieval-Augmented Generation (RAG) system that combines:
    
    - **LangGraph**: Workflow orchestration with four nodes (Plan → Retrieve → Answer → Reflect)
    - **ChromaDB**: Vector database for efficient document storage and retrieval
    - **Groq**: Fast LLM inference with Llama3/Mixtral models
    - **Comprehensive Evaluation**: Multiple metrics including ROUGE, BERTScore, and LLM-as-Judge
    
    ### 🏗️ Architecture
    
    1. **Plan Node**: Analyzes queries to determine if retrieval is needed
    2. **Retrieve Node**: Searches the vector database for relevant documents
    3. **Answer Node**: Generates responses using LLM and retrieved context
    4. **Reflect Node**: Evaluates answer quality and provides confidence scores
    
    ### 📊 Evaluation Metrics
    
    - **ROUGE Scores**: Text overlap similarity (ROUGE-1, ROUGE-2, ROUGE-L)
    - **BERTScore**: Semantic similarity using BERT embeddings
    - **LLM-as-Judge**: Quality assessment using the same LLM
    - **RAGAs**: Specialized RAG evaluation metrics (if available)
    
    ### 🚀 Features
    
    - Interactive web interface with real-time evaluation
    - Comprehensive evaluation framework with multiple metrics
    - Query history with export functionality
    - Dataset evaluation for systematic testing
    - Detailed processing logs and reflection
    - Confidence scoring and quality assessment
    
    ### 🔧 Configuration
    
    The system can be configured through environment variables:
    - `GROQ_API_KEY`: Your Groq API key
    - `LLM_MODEL`: Model to use (llama3-8b-8192, llama3-70b-8192, etc.)
    - `ENABLE_EVALUATION`: Enable/disable evaluation features
    - `LANGSMITH_API_KEY`: For advanced tracing (optional)
    """)

def export_history():
    """Export query history as JSON."""
    if 'query_history' not in st.session_state:
        st.warning("No history to export.")
        return
    
    # Prepare export data
    export_data = []
    for entry in st.session_state.query_history:
        export_entry = {
            'timestamp': entry['timestamp'],
            'query': entry['query'],
            'answer': entry['result'].answer,
            'confidence': entry['result'].confidence,
            'processing_time': entry['result'].processing_time,
            'retrieved_docs_count': len(entry['result'].retrieved_docs)
        }
        
        if 'evaluation' in entry:
            export_entry['evaluation'] = entry['evaluation']
        
        export_data.append(export_entry)
    
    # Create download
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 Download History (JSON)",
        data=json_str,
        file_name=f"rag_query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Q&A Agent</h1>', unsafe_allow_html=True)
    st.markdown("**Retrieval-Augmented Generation with Comprehensive Evaluation**")
    
    # Check if modules are available
    if not MODULES_AVAILABLE:
        st.error("❌ Required modules not available!")
        st.info("Please install dependencies: `pip install -r requirements.txt`")
        st.stop()
    
    # Quick setup info
    import os
    if not os.getenv("GROQ_API_KEY"):
        st.info("💡 **Quick Setup:** Add your Groq API key in the sidebar to get started!")
        st.markdown("1. Get a free API key from [console.groq.com](https://console.groq.com/)")
        st.markdown("2. Enter it in the sidebar under 'API Keys'")
        st.markdown("3. Start asking questions!")
        st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🤖 Q&A Interface", "🔬 Evaluation", "ℹ️ About"])
    
    with tab1:
        qa_interface()
    
    with tab2:
        evaluation_interface()
    
    with tab3:
        about_page()

if __name__ == "__main__":
    main()