# src/embeddings.py
import torch
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings  # Adjust the import path based on where you put the abstract class
import sys, os, toml
from src.utils import get_logger

# Get logger through the utility function
logger = get_logger(__name__)
import toml
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.toml'))
config = toml.load(config_path)


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible embeddings using SentenceTransformer."""
    
    def __init__(self):
        """Initialize the embedding model with configuration from config.toml."""
        self.model_path = config['embeddings']['model_path']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing SentenceTransformer with model {self.model_path} on {self.device}")
        
        try:
            self.model = SentenceTransformer(self.model_path, device=self.device)
            logger.info("SentenceTransformer model successfully loaded")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer model: {str(e)}")
            raise RuntimeError(f"Failed to initialize SentenceTransformer model: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to embed documents: {str(e)}")
            raise RuntimeError(f"Failed to embed documents: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        try:
            embedding = self.model.encode([text], normalize_embeddings=True)[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            raise RuntimeError(f"Failed to embed query: {str(e)}")

def get_embedding_function() -> Embeddings:
    """
    Initializes and returns a SentenceTransformer-based embedding model with GPU support if available.
    
    Returns:
        Embeddings: Configured embedding model compatible with LangChain
        
    Raises:
        RuntimeError: If model initialization fails
    """
    try:
        embeddings = SentenceTransformerEmbeddings()
        logger.info(f"Successfully created embedding function using model: {embeddings.model_path}")
        return embeddings
    except Exception as e:
        logger.error(f"Error creating embedding function: {e}")
        raise


print(get_embedding_function())