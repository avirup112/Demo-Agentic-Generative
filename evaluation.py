"""Evaluation framework for RAG Q&A Agent using multiple metrics."""

import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.info("RAGAs not available. This is optional for basic functionality.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.info("ROUGE not available. This is optional for basic functionality.")

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    logging.info("BERTScore not available. This is optional for basic functionality.")

from models import QueryResult
from rag_agent import RAGAgent
from config import Config

logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) if ROUGE_AVAILABLE else None
    
    def calculate_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        if not ROUGE_AVAILABLE or not self.rouge_scorer:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"ROUGE calculation failed: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def calculate_bert_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore."""
        if not BERT_SCORE_AVAILABLE:
            return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}
        
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            return {
                "bert_precision": P.mean().item(),
                "bert_recall": R.mean().item(),
                "bert_f1": F1.mean().item()
            }
        except Exception as e:
            logger.error(f"BERTScore calculation failed: {e}")
            return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}
    
    def calculate_llm_as_judge(self, agent: RAGAgent, question: str, answer: str, reference: str) -> Dict[str, float]:
        """Use LLM as a judge to evaluate answer quality."""
        try:
            judge_prompt = f"""
            Please evaluate the quality of the given answer compared to the reference answer.
            
            Question: {question}
            
            Generated Answer: {answer}
            
            Reference Answer: {reference}
            
            Please rate the generated answer on the following criteria (scale 1-5):
            1. Relevance: How well does the answer address the question?
            2. Accuracy: How factually correct is the answer compared to the reference?
            3. Completeness: How complete is the answer?
            4. Clarity: How clear and well-structured is the answer?
            
            Provide your ratings in this exact format:
            RELEVANCE: [1-5]
            ACCURACY: [1-5]
            COMPLETENESS: [1-5]
            CLARITY: [1-5]
            OVERALL: [1-5]
            """
            
            # Use the agent's LLM service for evaluation
            evaluation_result = agent.llm_service.generate_answer(judge_prompt, "")
            
            # Parse the evaluation result
            scores = self._parse_llm_judge_scores(evaluation_result)
            return scores
            
        except Exception as e:
            logger.error(f"LLM-as-Judge evaluation failed: {e}")
            return {
                "relevance": 3.0,
                "accuracy": 3.0,
                "completeness": 3.0,
                "clarity": 3.0,
                "overall": 3.0
            }
    
    def _parse_llm_judge_scores(self, evaluation_text: str) -> Dict[str, float]:
        """Parse LLM judge evaluation scores."""
        scores = {
            "relevance": 3.0,
            "accuracy": 3.0,
            "completeness": 3.0,
            "clarity": 3.0,
            "overall": 3.0
        }
        
        try:
            lines = evaluation_text.upper().split('\n')
            for line in lines:
                if 'RELEVANCE:' in line:
                    scores["relevance"] = float(line.split(':')[1].strip())
                elif 'ACCURACY:' in line:
                    scores["accuracy"] = float(line.split(':')[1].strip())
                elif 'COMPLETENESS:' in line:
                    scores["completeness"] = float(line.split(':')[1].strip())
                elif 'CLARITY:' in line:
                    scores["clarity"] = float(line.split(':')[1].strip())
                elif 'OVERALL:' in line:
                    scores["overall"] = float(line.split(':')[1].strip())
        except Exception as e:
            logger.error(f"Failed to parse LLM judge scores: {e}")
        
        return scores

class RAGEvaluator:
    """Main evaluation class for RAG Q&A Agent."""
    
    def __init__(self, agent: RAGAgent):
        self.agent = agent
        self.metrics = EvaluationMetrics()
        
    def evaluate_single_query(self, question: str, reference_answer: str, 
                            reference_contexts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate a single query against reference answer."""
        
        # Get agent response
        result = self.agent.query(question)
        
        # Calculate various metrics
        evaluation_results = {
            "question": question,
            "generated_answer": result.answer,
            "reference_answer": reference_answer,
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "retrieved_docs_count": len(result.retrieved_docs)
        }
        
        # ROUGE scores
        rouge_scores = self.metrics.calculate_rouge_scores(result.answer, reference_answer)
        evaluation_results.update(rouge_scores)
        
        # LLM-as-Judge scores
        llm_judge_scores = self.metrics.calculate_llm_as_judge(
            self.agent, question, result.answer, reference_answer
        )
        evaluation_results.update(llm_judge_scores)
        
        # RAGAs evaluation (if available and contexts provided)
        if RAGAS_AVAILABLE and reference_contexts:
            ragas_scores = self._evaluate_with_ragas(
                question, result.answer, reference_answer, 
                [doc.content for doc in result.retrieved_docs], reference_contexts
            )
            evaluation_results.update(ragas_scores)
        
        return evaluation_results
    
    def evaluate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Evaluate the agent on a dataset of questions and reference answers."""
        
        # Load dataset
        dataset = self._load_evaluation_dataset(dataset_path)
        
        if not dataset:
            logger.error("Failed to load evaluation dataset")
            return {}
        
        logger.info(f"Evaluating on {len(dataset)} questions...")
        
        all_results = []
        predictions = []
        references = []
        
        for i, item in enumerate(dataset):
            logger.info(f"Evaluating question {i+1}/{len(dataset)}")
            
            result = self.evaluate_single_query(
                item["question"],
                item["reference_answer"],
                item.get("reference_contexts")
            )
            
            all_results.append(result)
            predictions.append(result["generated_answer"])
            references.append(result["reference_answer"])
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(all_results, predictions, references)
        
        return {
            "individual_results": all_results,
            "aggregate_metrics": aggregate_metrics,
            "dataset_size": len(dataset)
        }
    
    def _evaluate_with_ragas(self, question: str, answer: str, reference: str,
                           contexts: List[str], reference_contexts: List[str]) -> Dict[str, float]:
        """Evaluate using RAGAs metrics."""
        if not RAGAS_AVAILABLE:
            return {}
        
        try:
            # Prepare data for RAGAs
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [reference]
            }
            
            dataset = Dataset.from_dict(data)
            
            # Run RAGAs evaluation
            result = evaluate(
                dataset,
                metrics=[
                    answer_relevancy,
                    faithfulness,
                    context_precision,
                    context_recall,
                    answer_correctness,
                    answer_similarity
                ]
            )
            
            return {
                "ragas_answer_relevancy": result["answer_relevancy"][0] if "answer_relevancy" in result else 0.0,
                "ragas_faithfulness": result["faithfulness"][0] if "faithfulness" in result else 0.0,
                "ragas_context_precision": result["context_precision"][0] if "context_precision" in result else 0.0,
                "ragas_context_recall": result["context_recall"][0] if "context_recall" in result else 0.0,
                "ragas_answer_correctness": result["answer_correctness"][0] if "answer_correctness" in result else 0.0,
                "ragas_answer_similarity": result["answer_similarity"][0] if "answer_similarity" in result else 0.0
            }
            
        except Exception as e:
            logger.error(f"RAGAs evaluation failed: {e}")
            return {}
    
    def _load_evaluation_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load evaluation dataset from JSON file."""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Dataset file not found: {dataset_path}")
            # Create a sample dataset
            return self._create_sample_dataset()
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return []
    
    def _create_sample_dataset(self) -> List[Dict[str, Any]]:
        """Create a sample evaluation dataset."""
        return [
            {
                "question": "What are the benefits of renewable energy?",
                "reference_answer": "Renewable energy offers environmental benefits like reduced greenhouse gas emissions, economic benefits including job creation and energy independence, and social benefits such as improved public health and energy access in remote areas.",
                "reference_contexts": [
                    "Renewable energy sources provide environmental, economic, and social benefits including reduced emissions and job creation."
                ]
            },
            {
                "question": "How does machine learning work?",
                "reference_answer": "Machine learning works by training algorithms on data to identify patterns and make predictions. It involves feeding large amounts of data to algorithms that learn to recognize patterns and make decisions without being explicitly programmed for each specific task.",
                "reference_contexts": [
                    "Machine learning algorithms learn patterns from data to make predictions and decisions automatically."
                ]
            },
            {
                "question": "What causes climate change?",
                "reference_answer": "Climate change is primarily caused by human activities that increase greenhouse gas concentrations in the atmosphere, particularly burning fossil fuels, deforestation, and industrial processes that release carbon dioxide, methane, and other greenhouse gases.",
                "reference_contexts": [
                    "Climate change is mainly caused by human activities that increase greenhouse gas emissions, especially from fossil fuel combustion."
                ]
            }
        ]
    
    def _calculate_aggregate_metrics(self, results: List[Dict[str, Any]], 
                                   predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate aggregate metrics across all evaluations."""
        
        # Basic statistics
        confidences = [r["confidence"] for r in results]
        processing_times = [r["processing_time"] for r in results]
        
        # ROUGE scores
        rouge1_scores = [r.get("rouge1", 0.0) for r in results]
        rouge2_scores = [r.get("rouge2", 0.0) for r in results]
        rougeL_scores = [r.get("rougeL", 0.0) for r in results]
        
        # LLM-as-Judge scores
        relevance_scores = [r.get("relevance", 0.0) for r in results]
        accuracy_scores = [r.get("accuracy", 0.0) for r in results]
        overall_scores = [r.get("overall", 0.0) for r in results]
        
        # BERTScore (calculated on all predictions at once)
        bert_scores = self.metrics.calculate_bert_score(predictions, references)
        
        aggregate = {
            "avg_confidence": np.mean(confidences),
            "avg_processing_time": np.mean(processing_times),
            "avg_rouge1": np.mean(rouge1_scores),
            "avg_rouge2": np.mean(rouge2_scores),
            "avg_rougeL": np.mean(rougeL_scores),
            "avg_relevance": np.mean(relevance_scores),
            "avg_accuracy": np.mean(accuracy_scores),
            "avg_overall_quality": np.mean(overall_scores),
            **bert_scores
        }
        
        # Add RAGAs metrics if available
        if RAGAS_AVAILABLE:
            ragas_metrics = [
                "ragas_answer_relevancy", "ragas_faithfulness", "ragas_context_precision",
                "ragas_context_recall", "ragas_answer_correctness", "ragas_answer_similarity"
            ]
            
            for metric in ragas_metrics:
                scores = [r.get(metric, 0.0) for r in results if metric in r]
                if scores:
                    aggregate[f"avg_{metric}"] = np.mean(scores)
        
        return aggregate
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""
        
        if not results:
            return "No evaluation results available."
        
        aggregate = results.get("aggregate_metrics", {})
        individual = results.get("individual_results", [])
        
        report = f"""
# RAG Q&A Agent Evaluation Report

## Summary
- Dataset Size: {results.get('dataset_size', 0)} questions
- Average Confidence: {aggregate.get('avg_confidence', 0):.3f}
- Average Processing Time: {aggregate.get('avg_processing_time', 0):.3f}s

## Text Similarity Metrics
- ROUGE-1: {aggregate.get('avg_rouge1', 0):.3f}
- ROUGE-2: {aggregate.get('avg_rouge2', 0):.3f}
- ROUGE-L: {aggregate.get('avg_rougeL', 0):.3f}
- BERTScore F1: {aggregate.get('bert_f1', 0):.3f}

## LLM-as-Judge Metrics (1-5 scale)
- Relevance: {aggregate.get('avg_relevance', 0):.2f}
- Accuracy: {aggregate.get('avg_accuracy', 0):.2f}
- Overall Quality: {aggregate.get('avg_overall_quality', 0):.2f}
"""

        # Add RAGAs metrics if available
        if any(key.startswith('avg_ragas_') for key in aggregate.keys()):
            report += "\n## RAGAs Metrics\n"
            ragas_metrics = {k: v for k, v in aggregate.items() if k.startswith('avg_ragas_')}
            for metric, value in ragas_metrics.items():
                clean_name = metric.replace('avg_ragas_', '').replace('_', ' ').title()
                report += f"- {clean_name}: {value:.3f}\n"
        
        # Add individual results summary
        if individual:
            report += f"\n## Individual Results Summary\n"
            report += f"- Best ROUGE-1: {max(r.get('rouge1', 0) for r in individual):.3f}\n"
            report += f"- Worst ROUGE-1: {min(r.get('rouge1', 0) for r in individual):.3f}\n"
            report += f"- Best Overall Quality: {max(r.get('overall', 0) for r in individual):.1f}/5\n"
            report += f"- Worst Overall Quality: {min(r.get('overall', 0) for r in individual):.1f}/5\n"
        
        return report

# Utility functions for testing
def create_sample_evaluation_dataset():
    """Create a sample evaluation dataset file."""
    dataset = [
        {
            "question": "What are the benefits of renewable energy?",
            "reference_answer": "Renewable energy offers environmental benefits like reduced greenhouse gas emissions, economic benefits including job creation and energy independence, and social benefits such as improved public health and energy access in remote areas.",
            "reference_contexts": [
                "Renewable energy sources provide environmental, economic, and social benefits including reduced emissions and job creation."
            ]
        },
        {
            "question": "How does machine learning work?",
            "reference_answer": "Machine learning works by training algorithms on data to identify patterns and make predictions. It involves feeding large amounts of data to algorithms that learn to recognize patterns and make decisions without being explicitly programmed for each specific task.",
            "reference_contexts": [
                "Machine learning algorithms learn patterns from data to make predictions and decisions automatically."
            ]
        },
        {
            "question": "What causes climate change?",
            "reference_answer": "Climate change is primarily caused by human activities that increase greenhouse gas concentrations in the atmosphere, particularly burning fossil fuels, deforestation, and industrial processes that release carbon dioxide, methane, and other greenhouse gases.",
            "reference_contexts": [
                "Climate change is mainly caused by human activities that increase greenhouse gas emissions, especially from fossil fuel combustion."
            ]
        },
        {
            "question": "What is artificial intelligence?",
            "reference_answer": "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, perception, and decision-making.",
            "reference_contexts": [
                "AI is computer science focused on creating intelligent machines that can perform human-like cognitive tasks."
            ]
        },
        {
            "question": "How does solar energy work?",
            "reference_answer": "Solar energy works by converting sunlight into electricity using photovoltaic cells or solar panels. When sunlight hits the solar cells, it creates an electric field that generates direct current (DC) electricity, which is then converted to alternating current (AC) for use in homes and businesses.",
            "reference_contexts": [
                "Solar panels convert sunlight into electricity through photovoltaic cells that generate electrical current when exposed to light."
            ]
        }
    ]
    
    with open("evaluation_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print("✅ Sample evaluation dataset created: evaluation_dataset.json")

def test_evaluation():
    """Test the evaluation framework."""
    print("🧪 Testing Evaluation Framework...")
    
    # Create sample dataset
    create_sample_evaluation_dataset()
    
    # Initialize agent and evaluator
    from rag_agent import RAGAgent
    
    agent = RAGAgent(initialize_kb=True)
    evaluator = RAGEvaluator(agent)
    
    # Test single query evaluation
    print("\n📝 Testing single query evaluation...")
    result = evaluator.evaluate_single_query(
        "What are the benefits of renewable energy?",
        "Renewable energy provides environmental and economic benefits including reduced emissions and job creation."
    )
    
    print(f"Generated answer: {result['generated_answer'][:100]}...")
    print(f"ROUGE-1: {result.get('rouge1', 0):.3f}")
    print(f"Overall quality: {result.get('overall', 0):.1f}/5")
    
    # Test dataset evaluation
    print("\n📊 Testing dataset evaluation...")
    dataset_results = evaluator.evaluate_dataset("evaluation_dataset.json")
    
    # Generate report
    report = evaluator.generate_evaluation_report(dataset_results)
    print(report)
    
    # Save results
    evaluator.save_evaluation_results(dataset_results, "evaluation_results.json")
    
    print("✅ Evaluation framework test completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_evaluation()