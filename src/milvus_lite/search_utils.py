"""
Search utilities for querying Milvus.
"""
import sys
import os

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from pymilvus import MilvusClient
from typing import List, Dict, Any, Optional
from src.utils import info, error


def search_vectors(
    client: MilvusClient,
    collection_name: str,
    query_vectors: List[List[float]],
    limit: int = 1,
    output_fields: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search vectors in Milvus collection with optional filtering.

    Args:
        client (MilvusClient): Milvus client.
        collection_name (str): Collection to search.
        query_vectors (List[List[float]]): Query vectors.
        limit (int): Max results to return.
        output_fields (Optional[List[str]]): Fields to return.
        filter_expr (Optional[str]): Filter expression (e.g. "subject == 'biology'").

    Returns:
        List[Dict[str, Any]]: Search results.
    """
    try:
        info(f"Running search on collection '{collection_name}' with limit={limit} and filter='{filter_expr}'", service="search_utils")
        results = client.search(
            collection_name=collection_name,
            data=query_vectors,
            limit=limit,
            output_fields=output_fields or [],
            filter=filter_expr
        )
        info(f"Search returned {len(results)} results", service="search_utils")
        return results
    except Exception as exc:
        error(f"Search failed on collection '{collection_name}': {exc}", service="search_utils")
        raise
