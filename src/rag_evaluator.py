"""
RAG Evaluation module for assessing system performance.
Q4: Évaluation du système RAG/LLM
"""
from typing import List, Dict, Any, Optional, Iterable
import logging
from dataclasses import dataclass
import re

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config_loader import ConfigLoader
from .qa_system import QASystem
from .document_retriever import DocumentRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetric:
    """Data class for evaluation metrics."""
    name: str
    score: float
    reasoning: Optional[str] = None


class RAGEvaluator:
    """
    Evaluates the performance of the RAG/LLM system.
    
    Assesses:
    - Answer relevance to the question
    - Faithfulness to the source documents
    - Answer completeness and quality
    """
    
    def __init__(self, config: ConfigLoader, qa_system: QASystem = None):
        """
        Initialize the RAG Evaluator.
        
        Args:
            config: ConfigLoader instance with system configuration
            qa_system: Optional QASystem instance
        """
        self.config = config
        self.qa_system = qa_system
        self.metrics = self.config.get('evaluation.metrics', ['relevance'])
        embedding_model = self.config.get('embedding_model.model_name')
        embedding_device = self.config.get('embedding_model.device', 'cpu')
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': embedding_device},
            encode_kwargs={'normalize_embeddings': True}
        )
        if qa_system and getattr(qa_system, 'retriever', None):
            self._retriever = qa_system.retriever
        else:
            try:
                self._retriever = DocumentRetriever(config)
            except Exception as exc:
                logger.debug(f"Unable to initialize retriever for evaluator: {exc}")
                self._retriever = None
    
    def evaluate_answer_relevance(
        self, 
        question: str, 
        answer: str
    ) -> EvaluationMetric:
        """
        Evaluate how relevant the answer is to the question.
        
        Args:
            question: The user's question
            answer: The generated answer
            
        Returns:
            EvaluationMetric with relevance score
        """
        def _tokenize(text: str) -> List[str]:
            return re.findall(r"[a-zA-Z0-9']+", text.lower())

        question_tokens = set(_tokenize(question))
        answer_tokens = set(_tokenize(answer))

        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when',
            'where', 'according', 'to', 'does', 'do', 'be', 'into', 'from', 'of', 'and'
        }
        question_tokens -= stop_words
        answer_tokens -= stop_words

        keyword_overlap = 0.0
        if question_tokens:
            overlap = len(question_tokens.intersection(answer_tokens))
            keyword_overlap = overlap / len(question_tokens)
        else:
            keyword_overlap = 0.3

        semantic_score = 0.0
        try:
            q_vec = self._embedding_model.embed_query(question)
            a_vec = self._embedding_model.embed_query(answer)
            dot = sum(q * a for q, a in zip(q_vec, a_vec))
            semantic_score = max(min((dot + 1.0) / 2.0, 1.0), 0.0)
        except Exception as exc:
            logger.debug(f"Embedding relevance fallback due to error: {exc}")
            semantic_score = keyword_overlap

        context_overlap = 0.0
        if answer_tokens and self._retriever is not None:
            try:
                raw_docs = self._retriever.search(question, top_k=3, return_scores=False)
                doc_tokens = set()
                for doc in raw_docs:
                    doc_tokens.update(_tokenize(doc.page_content))
                doc_tokens -= stop_words
                if doc_tokens:
                    context_overlap = len(answer_tokens.intersection(doc_tokens)) / max(len(answer_tokens), 1)
            except Exception as exc:
                logger.debug(f"Context overlap fallback due to error: {exc}")
                context_overlap = 0.0

        score = (
            0.3 * keyword_overlap +
            0.5 * semantic_score +
            0.2 * context_overlap
        )
        
        # Additional checks
        if "I don't have enough information" in answer or "Error:" in answer:
            score *= 0.3  # Penalize non-answers
        
        if len(answer) < 20:
            score *= 0.5  # Penalize very short answers
        
        reasoning = (
            f"Keyword overlap: {keyword_overlap:.2f}, "
            f"Semantic similarity: {semantic_score:.2f}, "
            f"Context overlap: {context_overlap:.2f}"
        )
        
        return EvaluationMetric(
            name="relevance",
            score=score,
            reasoning=reasoning
        )
    
    def _normalize_documents(
        self,
        source_documents: Optional[Iterable[Any]]
    ) -> List[Document]:
        """Convert heterogeneous document inputs into Document instances."""
        normalized: List[Document] = []

        if not source_documents:
            return normalized

        for item in source_documents:
            if isinstance(item, Document):
                normalized.append(item)
            elif isinstance(item, tuple) and item:
                doc_candidate = item[0]
                if isinstance(doc_candidate, Document):
                    normalized.append(doc_candidate)
            elif isinstance(item, dict):
                content = item.get("page_content") or item.get("content") or ""
                metadata = item.get("metadata", {})
                normalized.append(Document(page_content=content, metadata=metadata))

        return normalized

    def evaluate_faithfulness(
        self, 
        answer: str, 
        source_documents: Iterable[Any]
    ) -> EvaluationMetric:
        """
        Evaluate how faithful the answer is to the source documents.
        
        Args:
            answer: The generated answer
            source_documents: List of source documents used
            
        Returns:
            EvaluationMetric with faithfulness score
        """
        documents = self._normalize_documents(source_documents)

        if not documents:
            return EvaluationMetric(
                name="faithfulness",
                score=0.0,
                reasoning="No source documents provided"
            )
        
        # Concatenate all source content
        source_text = " ".join([doc.page_content.lower() for doc in documents])
        answer_lower = answer.lower()
        
        # Split answer into sentences/phrases
        answer_phrases = [p.strip() for p in answer.split('.') if len(p.strip()) > 10]
        
        if not answer_phrases:
            return EvaluationMetric(
                name="faithfulness",
                score=0.5,
                reasoning="Answer too short to evaluate"
            )
        
        # Check how many phrases have support in sources
        supported_phrases = 0
        for phrase in answer_phrases:
            # Check if key words from phrase appear in source
            phrase_words = set(phrase.split()) - {'the', 'a', 'an', 'is', 'are', 'was', 'were'}
            if phrase_words and any(word in source_text for word in phrase_words):
                supported_phrases += 1
        
        score = supported_phrases / len(answer_phrases) if answer_phrases else 0.0
        reasoning = f"{supported_phrases}/{len(answer_phrases)} phrases supported by sources"
        
        return EvaluationMetric(
            name="faithfulness",
            score=score,
            reasoning=reasoning
        )
    
    def evaluate_answer_completeness(
        self, 
        answer: str
    ) -> EvaluationMetric:
        """
        Evaluate the completeness and quality of the answer.
        
        Args:
            answer: The generated answer
            
        Returns:
            EvaluationMetric with completeness score
        """
        score = 0.0
        reasons = []
        
        # Length check
        if len(answer) >= 50:
            score += 0.3
            reasons.append("adequate length")
        elif len(answer) >= 30:
            score += 0.2
            reasons.append("moderate length")
        else:
            reasons.append("short answer")
        
        # Structure check
        if '.' in answer or '\n' in answer:
            score += 0.2
            reasons.append("structured")
        
        # Content check
        if not any(phrase in answer for phrase in ["I don't", "don't have", "Error:", "cannot"]):
            score += 0.3
            reasons.append("provides information")
        
        # Specificity check
        if any(char.isdigit() for char in answer) or any(word.isupper() for word in answer.split()):
            score += 0.2
            reasons.append("specific details")
        
        reasoning = ", ".join(reasons)
        
        return EvaluationMetric(
            name="completeness",
            score=min(score, 1.0),
            reasoning=reasoning
        )
    
    def evaluate_response(
        self, 
        question: str,
        answer: str,
        source_documents: Optional[List[Document]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a QA response.
        
        Args:
            question: The user's question
            answer: The generated answer
            source_documents: Optional list of source documents
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info(f"Evaluating response for question: {question}")
        
        metrics = {}
        
        # Evaluate relevance
        relevance = self.evaluate_answer_relevance(question, answer)
        metrics['relevance'] = {
            'score': relevance.score,
            'reasoning': relevance.reasoning
        }
        
        # Evaluate faithfulness if sources provided
        if source_documents:
            faithfulness = self.evaluate_faithfulness(answer, source_documents)
            metrics['faithfulness'] = {
                'score': faithfulness.score,
                'reasoning': faithfulness.reasoning
            }
        
        # Evaluate completeness
        completeness = self.evaluate_answer_completeness(answer)
        metrics['completeness'] = {
            'score': completeness.score,
            'reasoning': completeness.reasoning
        }
        
        # Calculate overall score
        scores = [m['score'] for m in metrics.values()]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        result = {
            'question': question,
            'answer': answer,
            'metrics': metrics,
            'overall_score': overall_score,
            'grade': self._get_grade(overall_score)
        }
        
        logger.info(f"Overall score: {overall_score:.2f}")
        
        return result
    
    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def evaluate_qa_result(self, qa_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a result from QASystem.answer_question().
        
        Args:
            qa_result: Result dictionary from QASystem
            
        Returns:
            Evaluation result dictionary
        """
        question = qa_result.get('question', '')
        answer = qa_result.get('answer', '')
        
        # Extract source documents if available
        source_docs = None
        if 'sources' in qa_result:
            source_docs = [
                Document(page_content=s['content'], metadata=s.get('metadata', {}))
                for s in qa_result['sources']
            ]
        
        return self.evaluate_response(question, answer, source_docs)
    
    def print_evaluation(self, evaluation: Dict[str, Any]):
        """
        Print evaluation results in a formatted way.
        
        Args:
            evaluation: Evaluation result dictionary
        """
        print(f"\n{'='*80}")
        print(f"EVALUATION REPORT")
        print(f"{'='*80}\n")
        
        print(f"Question: {evaluation['question']}\n")
        print(f"Answer: {evaluation['answer']}\n")
        
        print(f"{'='*80}")
        print(f"METRICS:")
        print(f"{'='*80}\n")
        
        for metric_name, metric_data in evaluation['metrics'].items():
            print(f"{metric_name.upper()}:")
            print(f"  Score: {metric_data['score']:.2f}")
            print(f"  Reasoning: {metric_data['reasoning']}\n")
        
        print(f"{'='*80}")
        print(f"OVERALL SCORE: {evaluation['overall_score']:.2f} (Grade: {evaluation['grade']})")
        print(f"{'='*80}\n")
    
    def batch_evaluate(
        self, 
        qa_results: List[Dict[str, Any]],
        print_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple QA results.
        
        Args:
            qa_results: List of QA result dictionaries
            print_results: Whether to print results
            
        Returns:
            List of evaluation dictionaries
        """
        evaluations = []
        
        for i, qa_result in enumerate(qa_results):
            print(f"\n{'#'*80}")
            print(f"Evaluation {i+1}/{len(qa_results)}")
            print(f"{'#'*80}")
            
            evaluation = self.evaluate_qa_result(qa_result)
            evaluations.append(evaluation)
            
            if print_results:
                self.print_evaluation(evaluation)
        
        # Print summary
        if evaluations:
            avg_score = sum(e['overall_score'] for e in evaluations) / len(evaluations)
            print(f"\n{'='*80}")
            print(f"SUMMARY: Average Overall Score: {avg_score:.2f}")
            print(f"{'='*80}\n")
        
        return evaluations
