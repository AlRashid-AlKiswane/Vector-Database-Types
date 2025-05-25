"""
Configuration and client initialization for Milvus.
"""

import sys
import os

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from src.utils import info, error

# Constants
MILVUS_DB_PATH = "milvus_demo.db"
COLLECTION_NAME = "demo_collection"
VECTOR_DIM = 384


def get_milvus_client() -> MilvusClient:
    """
    Initialize and return MilvusClient.

    Returns:
        MilvusClient: Milvus client connected to the configured DB.
    """
    try:
        client = MilvusClient(MILVUS_DB_PATH)
        info(f"Milvus client initialized with DB path '{MILVUS_DB_PATH}'", service="config")
        return client
    except Exception as exc:
        error(f"Failed to initialize Milvus client: {exc}", service="config")
        raise


def get_embedding_model() -> SentenceTransformer:
    """
    Initialize and return the embedding model.

    Returns:
        SentenceTransformer: SentenceTransformer model instance.
    """
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        info("Embedding model 'all-MiniLM-L6-v2' loaded", service="config")
        return model
    except Exception as exc:
        error(f"Failed to load embedding model: {exc}", service="config")
        raise
