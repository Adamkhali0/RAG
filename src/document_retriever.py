"""
Document Retriever module for querying the vector database.
Q2: Recherche documentaire dans la base vectorielle
"""
from typing import List, Tuple, Dict, Any
import logging

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .config_loader import ConfigLoader
from .document_indexer import DocumentIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Handles document retrieval from the vector database.
    Performs similarity search and returns relevant documents with scores.
    """
    
    def __init__(self, config: ConfigLoader, vector_store: FAISS = None):
        """
        Initialize the DocumentRetriever.
        
        Args:
            config: ConfigLoader instance with system configuration
            vector_store: Optional Chroma vector store instance
        """
        self.config = config
        self.vector_store = vector_store
        self.top_k = self.config.get('retrieval.top_k', 4)
        self.search_type = self.config.get('retrieval.search_type', 'similarity')
        self.score_threshold = self.config.get('retrieval.score_threshold', 0.0)
        
        if self.vector_store is None:
            self._load_vector_store()
    
    def _load_vector_store(self):
        """Load the vector store using DocumentIndexer."""
        logger.info("Loading vector store for retrieval...")
        indexer = DocumentIndexer(self.config)
        self.vector_store = indexer.load_vector_store()
    
    def search(
        self, 
        query: str, 
        top_k: int = None,
        return_scores: bool = True
    ) -> List[Tuple[Document, float]] | List[Document]:
        """
        Search for documents relevant to the query.
        
        Args:
            query: User query string
            top_k: Number of documents to return (overrides config)
            return_scores: Whether to return similarity scores
            
        Returns:
            List of (Document, score) tuples if return_scores=True,
            otherwise list of Documents
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized")
        
        k = top_k if top_k is not None else self.top_k
        
        logger.info(f"Searching for query: '{query}' (top_k={k})")
        
        if return_scores:
            # Similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter by score threshold
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= self.score_threshold
            ]
            
            logger.info(f"Found {len(filtered_results)} documents above threshold")
            
            return filtered_results
        else:
            # Similarity search without scores
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            
            logger.info(f"Found {len(results)} documents")
            
            return results
    
    def search_with_relevance_scores(
        self, 
        query: str, 
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents with relevance scores (0-1 range, higher is better).
        
        Args:
            query: User query string
            top_k: Number of documents to return
            
        Returns:
            List of (Document, relevance_score) tuples
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized")
        
        k = top_k if top_k is not None else self.top_k
        
        logger.info(f"Searching with relevance scores: '{query}' (top_k={k})")
        
        results = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=k
        )
        
        logger.info(f"Found {len(results)} documents")
        
        return results
    
    def format_results(
        self, 
        results: List[Tuple[Document, float]],
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Format search results into a structured dictionary format.
        
        Args:
            results: List of (Document, score) tuples
            include_metadata: Whether to include document metadata
            
        Returns:
            List of formatted result dictionaries
        """
        formatted_results = []
        
        for i, (doc, score) in enumerate(results):
            result = {
                'rank': i + 1,
                'content': doc.page_content,
                'score': float(score),
            }
            
            if include_metadata:
                result['metadata'] = doc.metadata
            
            formatted_results.append(result)
        
        return formatted_results
    
    def print_results(
        self, 
        results: List[Tuple[Document, float]],
        max_content_length: int = 200
    ):
        """
        Print search results in a readable format.
        
        Args:
            results: List of (Document, score) tuples
            max_content_length: Maximum length of content to display
        """
        print(f"\n{'='*80}")
        print(f"Found {len(results)} relevant documents:")
        print(f"{'='*80}\n")
        
        for i, (doc, score) in enumerate(results):
            print(f"[{i+1}] Score: {score:.4f}")
            
            # Print metadata
            if doc.metadata:
                metadata_str = ", ".join([
                    f"{k}: {v}" for k, v in doc.metadata.items()
                    if k in ['filename', 'page', 'source']
                ])
                print(f"    Metadata: {metadata_str}")
            
            # Print content preview
            content = doc.page_content[:max_content_length]
            if len(doc.page_content) > max_content_length:
                content += "..."
            print(f"    Content: {content}")
            print(f"{'-'*80}\n")
    
    def get_relevant_context(
        self, 
        query: str, 
        top_k: int = None
    ) -> str:
        """
        Get relevant context as a single concatenated string.
        Useful for passing to LLM.
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve
            
        Returns:
            Concatenated context string
        """
        results = self.search(query, top_k=top_k, return_scores=False)
        
        context_parts = []
        for i, doc in enumerate(results):
            context_parts.append(f"[Document {i+1}]\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def test_queries(self, queries: List[str], top_k: int = None):
        """
        Test multiple queries and print results.
        
        Args:
            queries: List of query strings to test
            top_k: Number of documents to return per query
        """
        for i, query in enumerate(queries):
            print(f"\n{'#'*80}")
            print(f"Test Query {i+1}: {query}")
            print(f"{'#'*80}")
            
            results = self.search(query, top_k=top_k, return_scores=True)
            self.print_results(results)
