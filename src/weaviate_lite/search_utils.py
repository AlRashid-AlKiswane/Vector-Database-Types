"""
Search functionality for Weaviate.

This module provides a function to perform vector similarity search
in a Weaviate vector database.
"""

import sys
import os
from typing import List

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from src.utils.logging_utils import info, error


def search_documents(
    client,
    query_vector: List[float],
    class_name: str = "Document",
    limit: int = 2
) -> dict:
    """
    Perform a vector similarity search in the Weaviate database.

    Args:
        client (weaviate.Client): The initialized Weaviate client.
        query_vector (List[float]): The vector to search against.
        class_name (str): The class name to search in. Default is "Document".
        limit (int): The number of top results to return. Default is 2.

    Returns:
        dict: The raw response dictionary from Weaviate.

    Raises:
        Exception: If the search fails.
    """
    try:
        results = (
            client.query
            .get(class_name, ["text"])
            .with_near_vector({"vector": query_vector})
            .with_limit(limit)
            .do()
        )
        hits = results.get("data", {}).get("Get", {}).get(class_name, [])
        info("🔍 Search returned {} result(s) from '{}'".format(len(hits), class_name))
        return results
    except Exception as exc:
        error("❌ Search failed: {}".format(str(exc)))
        raise
