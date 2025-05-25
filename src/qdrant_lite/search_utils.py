"""Search functionality using Qdrant"""
import sys
import os

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from src.utils.logging_utils import info, error
from typing import List

def search_qdrant(client, collection_name: str, query: List[float], limit: int = 2):
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query,
            limit=limit
        )
        info(f"🔍 Search completed with {len(results)} results")
        return results
    except Exception as e:
        error(f"❌ Failed to search: {e}")
        raise
