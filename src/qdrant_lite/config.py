"""Qdrant configuration setup"""
import sys
import os

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import VectorParams, Distance
from src.utils.logging_utils import info, error

COLLECTION_NAME = "demo_collection"
VECTOR_DIM = 384

def get_qdrant_client() -> QdrantClient:
    try:
        client = QdrantClient(":memory:")
        info("✅ Qdrant client initialized")
        return client
    except Exception as e:
        error(f"❌ Failed to initialize Qdrant client: {e}")
        raise

def get_embedding_model() -> SentenceTransformer:
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        info("📐 Embedding model loaded")
        return model
    except Exception as e:
        error(f"❌ Failed to load embedding model: {e}")
        raise
