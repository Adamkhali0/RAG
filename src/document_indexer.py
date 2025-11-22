"""
Document Indexer module for loading, splitting, embedding, and storing documents.
Q1: Mise en place d'un système d'indexation des documents
"""
from pathlib import Path
from typing import Dict, List, Optional
import logging

from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Handles the complete pipeline for document indexation:
    - Loading documents from various sources
    - Splitting documents into chunks
    - Computing embeddings
    - Storing in vector database
    """
    _EMBEDDINGS_CACHE: Dict[str, HuggingFaceEmbeddings] = {}
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize the DocumentIndexer.
        
        Args:
            config: ConfigLoader instance with system configuration
        """
        self.config = config
        self.embeddings = self._initialize_embeddings()
        self.vector_store = None
        self.text_splitter = self._initialize_text_splitter()
        
    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Initialize the embedding model from HuggingFace.
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        model_name = self.config.get('embedding_model.model_name')
        device = self.config.get('embedding_model.device', 'cpu')
        cache_key = f"{model_name}:{device}"
        
        if cache_key in DocumentIndexer._EMBEDDINGS_CACHE:
            logger.info(f"Reusing cached embedding model: {model_name} on {device}")
            return DocumentIndexer._EMBEDDINGS_CACHE[cache_key]

        logger.info(f"Initializing embedding model: {model_name} on {device}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Cache embeddings to avoid repeated downloads when multiple components instantiate the indexer.
        DocumentIndexer._EMBEDDINGS_CACHE[cache_key] = embeddings
        
        return embeddings
    
    def _initialize_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        Initialize the text splitter for document chunking.
        Optimized for Markdown format.
        
        Returns:
            Text splitter instance
        """
        chunk_size = self.config.get('document_processing.chunk_size', 1000)
        chunk_overlap = self.config.get('document_processing.chunk_overlap', 200)
        
        # Markdown-optimized separators
        separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            ", ",    # Clauses
            " ",     # Words
            ""       # Characters
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )
        
        logger.info(f"Text splitter initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        return text_splitter
    
    def load_documents(self, data_directory: Optional[str] = None) -> List[Document]:
        """
        Load documents from the specified directory.
        Supports PDF files with metadata preservation.
        
        Args:
            data_directory: Path to directory containing documents
            
        Returns:
            List of loaded Document objects with metadata
        """
        if data_directory is None:
            data_directory = self.config.get('document_processing.data_directory')
        
        data_path = Path(data_directory)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_directory}")
        
        logger.info(f"Loading documents from: {data_directory}")
        
        # Use PDFPlumberLoader which is more robust for complex PDFs
        pdf_loader = DirectoryLoader(
            str(data_path),
            glob="**/*.pdf",
            loader_cls=PDFPlumberLoader,
            show_progress=True,
            use_multithreading=False  # Avoid threading issues with complex PDFs
        )
        
        documents = pdf_loader.load()
        logger.info(f"Loaded {len(documents)} document pages")
        
        # Enhance metadata
        for doc in documents:
            if 'source' in doc.metadata:
                source_path = Path(doc.metadata['source'])
                doc.metadata['filename'] = source_path.name
                doc.metadata['directory'] = str(source_path.parent)
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks while preserving metadata.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of Document chunks with preserved metadata
        """
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        
        # Filter out empty documents
        non_empty_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        
        if len(non_empty_docs) < len(documents):
            logger.warning(f"Filtered out {len(documents) - len(non_empty_docs)} empty documents")
        
        if not non_empty_docs:
            logger.error("No documents with content found!")
            return []
        
        chunks = self.text_splitter.split_documents(non_empty_docs)

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

        cleaned_chunks: List[Document] = []
        for chunk in chunks:
            lines = chunk.page_content.splitlines()
            filtered_lines = []
            for line in lines:
                upper_line = line.upper()
                if any(marker in upper_line for marker in skip_markers):
                    continue
                if upper_line.strip().startswith("LO "):
                    continue
                filtered_lines.append(line)
            cleaned_text = "\n".join(filtered_lines).strip()
            if not cleaned_text:
                continue
            chunk.page_content = cleaned_text
            cleaned_chunks.append(chunk)

        # Add chunk-specific metadata
        for i, chunk in enumerate(cleaned_chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        logger.info(f"Created {len(cleaned_chunks)} chunks")
        
        return cleaned_chunks
    
    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """
        Create and populate the vector store with document embeddings.
        Uses FAISS for better Windows compatibility.
        
        Args:
            chunks: List of document chunks to embed and store
            
        Returns:
            FAISS vector store instance
        """
        persist_directory = self.config.get('vector_store.persist_directory')
        
        logger.info(f"Creating FAISS vector store with {len(chunks)} chunks...")
        logger.info(f"Persist directory: {persist_directory}")
        
        if not chunks:
            raise ValueError("Cannot create vector store with 0 chunks. Check that PDFs contain extractable text.")
        
        # Create FAISS vector store from documents
        logger.info("Generating embeddings and building index...")
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Save to disk
        logger.info(f"Saving index to {persist_directory}...")
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(persist_directory)
        
        logger.info("FAISS vector store created successfully")
        
        return self.vector_store
    
    def load_vector_store(self) -> FAISS:
        """
        Load an existing vector store from disk.
        
        Returns:
            FAISS vector store instance
        """
        persist_directory = self.config.get('vector_store.persist_directory')
        
        logger.info(f"Loading FAISS vector store from: {persist_directory}")
        
        self.vector_store = FAISS.load_local(
            persist_directory,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        logger.info("FAISS vector store loaded successfully")
        
        return self.vector_store
    
    def index_documents(self, data_directory: Optional[str] = None) -> FAISS:
        """
        Complete pipeline: Load -> Split -> Embed -> Store documents.
        
        Args:
            data_directory: Path to directory containing documents
            
        Returns:
            Chroma vector store with indexed documents
        """
        logger.info("Starting document indexation pipeline...")
        
        # Step 1: Load documents
        documents = self.load_documents(data_directory)
        
        # Step 2: Split documents
        chunks = self.split_documents(documents)
        
        # Step 3 & 4: Embed and store
        vector_store = self.create_vector_store(chunks)
        
        logger.info("Document indexation completed successfully")
        
        return vector_store
    
    def get_vector_store(self) -> Optional[FAISS]:
        """
        Get the current vector store instance.
        
        Returns:
            Chroma vector store or None if not initialized
        """
        return self.vector_store
