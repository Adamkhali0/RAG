"""
Question-Answering System using LLM and RAG.
Q3: Système de question-réponse basé sur un LLM
"""
from typing import Dict, Any, Optional, List
import logging
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

from .config_loader import ConfigLoader
from .document_retriever import DocumentRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QASystem:
    """
    Question-Answering system that combines document retrieval with LLM.
    Uses retrieved context to generate accurate, grounded answers.
    """
    
    def __init__(self, config: ConfigLoader, retriever: DocumentRetriever = None):
        """
        Initialize the QA System.
        
        Args:
            config: ConfigLoader instance with system configuration
            retriever: Optional DocumentRetriever instance
        """
        self.config = config
        self.retriever = retriever if retriever else DocumentRetriever(config)
        self.tokenizer: Optional[AutoTokenizer] = None
        self.llm = self._initialize_llm()
        self.prompt_template = self._create_prompt_template()
    
    @staticmethod
    def _language_hint(text: str) -> str:
        """Return a short language hint based on accented French characters."""
        lowered = text.lower()
        french_chars = any(ch in lowered for ch in "éàèùçâêîôûïëœ")
        return "[French excerpt]" if french_chars else ""

    def _extract_relevant_snippet(
        self,
        text: str,
        question: Optional[str],
        max_length: int
    ) -> str:
        """Return a snippet biased toward sections matching the user's question."""
        if not text:
            return ""

        cleaned = text.strip()
        if not cleaned:
            return ""

        normalized = re.sub(r"\s+", " ", cleaned)
        if max_length <= 0 or len(normalized) <= max_length:
            return normalized

        if not question:
            return normalized[:max_length]

        tokens = [token for token in re.findall(r"\w+", question.lower()) if len(token) >= 4]
        if not tokens:
            return normalized[:max_length]

        lower_text = normalized.lower()
        window = max_length
        step = max(80, window // 2)
        best_start = 0
        best_score = -1

        for start in range(0, len(normalized), step):
            snippet = lower_text[start:start + window]
            score = sum(snippet.count(token) for token in tokens)
            if score > best_score:
                best_score = score
                best_start = start

        if best_score <= 0:
            return normalized[:max_length]

        return normalized[best_start:best_start + window]

    def _build_context_parts(
        self,
        documents: List[Document],
        max_document_length: int,
        question: Optional[str] = None
    ) -> List[str]:
        """Return formatted context blocks truncated to the desired length."""
        context_parts: List[str] = []
        skip_markers = (
            "PRACTICE QUIZ",
            "PRACTICE QUIZZES",
            "PRACTICE QUESTIONS",
            "MULTIPLE CHOICE",
            "TRUE/FALSE",
            "ANSWER KEY",
            "LEARNING OBJECTIVE",
            "LEARNING OBJECTIVES",
            "CHECK YOUR ANSWERS",
        )
        for idx, doc in enumerate(documents):
            if not doc.page_content:
                continue
            snippet = self._extract_relevant_snippet(
                doc.page_content,
                question,
                max_document_length
            ).strip()
            if not snippet:
                continue
            upper_snippet = snippet.upper()
            if any(marker in upper_snippet for marker in skip_markers):
                continue
            if re.search(r'\bLO\s*\d', snippet):
                continue
            snippet = snippet.replace('{{', '').replace('}}', '')
            snippet = re.sub(r'(?m)^\s*[A-Da-d][\)\.]\s+', '', snippet)
            snippet = re.sub(r'\s{3,}', '  ', snippet)
            hint = self._language_hint(snippet)
            header = f"Document {idx + 1}"
            if hint:
                header = f"{header} {hint}"
            context_parts.append(f"{header}:\n{snippet}")
        return context_parts

    @staticmethod
    def _trim_block(block: str) -> str:
        """Progressively shrink a context block while keeping its header."""
        if not block:
            return ""

        header, sep, body = block.partition("\n")
        if not sep:
            header = ""
            body = block

        if len(body) <= 80:
            return ""

        new_length = max(80, int(len(body) * 0.7))
        trimmed_body = body[:new_length].rstrip()

        if not trimmed_body:
            return ""

        return f"{header}\n{trimmed_body}" if header else trimmed_body

    def _prepare_prompt(self, question: str, context_parts: List[str]) -> tuple[str, str]:
        """Build the final prompt while keeping it under the token budget."""
        default_context = "No relevant context was retrieved."
        max_prompt_tokens = int(self.config.get('llm.max_prompt_tokens', 0) or 0)

        if not context_parts:
            context = default_context
            prompt = self.prompt_template.format(context=context, question=question)
            return context, prompt

        if not self.tokenizer or max_prompt_tokens <= 0:
            context = "\n\n".join(context_parts)
            prompt = self.prompt_template.format(context=context, question=question)
            return context, prompt

        parts = context_parts[:]

        while parts:
            context = "\n\n".join(parts)
            prompt = self.prompt_template.format(context=context, question=question)
            token_count = len(self.tokenizer.encode(prompt, add_special_tokens=False))

            if token_count <= max_prompt_tokens:
                return context, prompt

            trimmed = self._trim_block(parts[-1])
            if trimmed:
                parts[-1] = trimmed
            else:
                parts.pop()

        context = default_context
        prompt = self.prompt_template.format(context=context, question=question)
        return context, prompt

    def _initialize_llm(self) -> HuggingFacePipeline:
        """
        Initialize the open-source LLM from HuggingFace.
        
        Returns:
            HuggingFacePipeline instance
        """
        model_name = self.config.get('llm.model_name')
        temperature = self.config.get('llm.temperature', 0.7)
        max_length = self.config.get('llm.max_length', 512)
        max_new_tokens = self.config.get('llm.max_new_tokens', 160)
        do_sample = self.config.get('llm.do_sample', False)
        num_beams = self.config.get('llm.num_beams', 4)
        
        logger.info(f"Loading LLM: {model_name}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        self.tokenizer = tokenizer

        # Respect model's maximum sequence length and prefer left truncation to keep recent context
        model_max_length = getattr(tokenizer, "model_max_length", max_length)
        effective_max_length = min(max_length, model_max_length)
        self.tokenizer.model_max_length = effective_max_length
        self.tokenizer.truncation_side = "left"

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "no_repeat_ngram_size": 3,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.9
        else:
            # Deterministic decoding works better with a modest length penalty
            generation_kwargs["length_penalty"] = 0.9
        
        # Create pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            **generation_kwargs,
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        logger.info("LLM initialized successfully")
        
        return llm
    
    def _create_prompt_template(self) -> PromptTemplate:
        """
        Create an optimized prompt template for the QA system.
        
        The template includes:
        - Clear instructions for the LLM
        - Context from retrieved documents
        - The user's question
        - Guidelines for answer format
        
        Returns:
            PromptTemplate instance
        """
        template = """You are a helpful AI assistant that answers questions based on the provided context.

    Use the following pieces of context to answer the question at the end. 
    If you cannot find the answer in the context, say "I don't have enough information to answer this question."
    Do not make up information that is not in the context.

    Context:
    {context}

    Question: {question}

    Instructions:
    1. Provide a clear and concise answer based only on the context above
    2. If the context contains relevant information, synthesize it into a coherent response
    3. If multiple pieces of context are relevant, combine them in your answer
    4. Cite specific details from the context when possible
    5. If the answer is not in the context, clearly state that
    6. Answer in English even if the context is not in English
    7. Keep the answer under four sentences and avoid repeating raw matrix notation such as [A, B]T unless explicitly required
    8. Ignore multiple-choice labels (for example "a.", "b.") or templating braces ("{{"/"}}"); rewrite the information as plain sentences or a simple numbered list
    9. Present lists using natural language (e.g., "First...", "Second...") or numbered bullets, and ensure each sentence is grammatically complete
    10. If the question specifies a number of items, deliver exactly that number when the context provides them, otherwise acknowledge the gap
    11. When the context mentions both experiments and quasi-experiments, focus on the full experiment details unless the user explicitly asks about quasi-experiments

    Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        return prompt
    
    def answer_question(
        self, 
        question: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using the RAG system.
        
        Args:
            question: User's question
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary containing answer and optionally source documents
        """
        logger.info(f"Processing question: {question}")
        
        try:
            # Retrieve relevant documents
            search_results = self.retriever.search(question)

            # Handle both Document and (Document, score) tuple formats
            documents: List[Document] = []
            for item in search_results:
                doc = item[0] if isinstance(item, tuple) else item
                if isinstance(doc, Document):
                    documents.append(doc)

            max_context_docs = self.config.get('llm.max_context_docs', 3)
            max_document_length = self.config.get('llm.max_document_length', 400)
            limited_docs = documents[:max_context_docs]

            context_parts = self._build_context_parts(
                limited_docs,
                max_document_length,
                question
            )
            context, prompt = self._prepare_prompt(question, context_parts)

            # Generate answer using LLM
            raw_answer = self.llm.invoke(prompt)
            if isinstance(raw_answer, str):
                answer = raw_answer
            elif isinstance(raw_answer, list) and raw_answer:
                first = raw_answer[0]
                if isinstance(first, dict) and 'generated_text' in first:
                    answer = first['generated_text']
                else:
                    answer = str(first)
            elif isinstance(raw_answer, dict) and 'generated_text' in raw_answer:
                answer = raw_answer['generated_text']
            else:
                answer = str(raw_answer)
            
            answer = self.normalize_answer(answer)

            response = {
                "question": question,
                "answer": answer,
            }
            
            if return_sources:
                response["sources"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in limited_docs
                ]
                response["num_sources"] = len(limited_docs)
            
            logger.info("Question answered successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": []
            }

    @staticmethod
    def normalize_answer(text: str) -> str:
        """Clean up raw model output for readability."""
        if not isinstance(text, str):
            return str(text)

        cleaned = text.strip()
        cleaned = re.sub(r'(\[A,\s*B\]T\.?\s*){2,}', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(\.\s*){4,}', '...', cleaned)
        cleaned = re.sub(r'\{\{|\}\}', '', cleaned)
        cleaned = re.sub(r'(?m)^[A-Da-d][\)\.]\s*', '', cleaned)
        cleaned = re.sub(r'\s+[A-Da-d][\)\.]\s+', '. ', cleaned)
        cleaned = re.sub(r'(?m)^\s*[A-Da-d][\)\.]\s*', '', cleaned)
        cleaned = cleaned.replace('..', '.')
        cleaned = cleaned.replace(' and and ', ' and ')
        cleaned = re.sub(r'(?i)quasi[-\s]experiment[^.]*\.', '', cleaned)
        cleaned = re.sub(r'(?i)quasi[-\s]independent[^.]*\.', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)

        segments = [seg.strip() for seg in cleaned.split('.') if seg.strip()]
        if 2 <= len(segments) <= 5 and all(len(seg.split()) <= 8 for seg in segments):
            cleaned = '; '.join(segments) + '.'
        cleaned = cleaned.strip()
        return cleaned or "I don't have enough information to answer this question."
    
    def answer_with_custom_context(
        self, 
        question: str,
        context: str
    ) -> str:
        """
        Answer a question with custom provided context.
        
        Args:
            question: User's question
            context: Custom context string
            
        Returns:
            Generated answer
        """
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        answer = self.llm.invoke(prompt)
        
        return answer
    
    def print_answer(self, result: Dict[str, Any]):
        """
        Print the answer in a formatted way.
        
        Args:
            result: Result dictionary from answer_question
        """
        print(f"\n{'='*80}")
        print(f"Question: {result['question']}")
        print(f"{'='*80}\n")
        print(f"Answer: {result['answer']}\n")
        
        if "sources" in result and result["sources"]:
            print(f"{'='*80}")
            print(f"Sources ({result['num_sources']} documents):")
            print(f"{'='*80}\n")
            
            for i, source in enumerate(result["sources"]):
                print(f"[Source {i+1}]")
                if source["metadata"]:
                    metadata_str = ", ".join([
                        f"{k}: {v}" for k, v in source["metadata"].items()
                        if k in ['filename', 'page', 'source']
                    ])
                    print(f"Metadata: {metadata_str}")
                
                content_preview = source["content"][:300]
                if len(source["content"]) > 300:
                    content_preview += "..."
                print(f"Content: {content_preview}\n")
    
    def batch_answer(
        self, 
        questions: list[str],
        print_results: bool = True
    ) -> list[Dict[str, Any]]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of questions
            print_results: Whether to print results
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, question in enumerate(questions):
            print(f"\n{'#'*80}")
            print(f"Question {i+1}/{len(questions)}")
            print(f"{'#'*80}")
            
            result = self.answer_question(question)
            results.append(result)
            
            if print_results:
                self.print_answer(result)
        
        return results
