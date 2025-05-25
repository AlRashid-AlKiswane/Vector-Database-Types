"""Index creation and insertion utilities for Qdrant"""
import sys
import os

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from qdrant_client.models import VectorParams, Distance, PointStruct
from src.utils.logging_utils import info, error
from typing import List

def create_qdrant_collection(client, collection_name: str, vector_dim: int) -> None:
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        info(f"📦 Qdrant collection '{collection_name}' created")
    except Exception as e:
        error(f"❌ Failed to create collection: {e}")
        raise

def insert_data(client, collection_name: str, embeddings: List[List[float]], texts: List[str]) -> None:
    try:
        points = [PointStruct(id=i, vector=embeddings[i], payload={"text": texts[i]})
                  for i in range(len(texts))]
        client.upsert(collection_name=collection_name, points=points)
        info("✅ Data inserted into Qdrant collection")
    except Exception as e:
        error(f"❌ Failed to insert data: {e}")
        raise
