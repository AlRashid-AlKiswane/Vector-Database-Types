"""
Utility functions for performing searches and reranking using Pinecone.
"""
import sys
import os

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from src.utils import info


def basic_search(index, query: str, namespace: str) -> None:
    """
    Performs a basic vector search on the index.

    Args:
        index: Pinecone index object.
        query (str): Search query text.
        namespace (str): Namespace to search in.
    """
    info("🔎 Performing basic search for query: '%s'", query)

    result = index.search(
        namespace=namespace,
        query={
            "top_k": 5,
            "inputs": {
                "text": query
            }
        }
    )

    info("📄 Basic Search Results:")
    info(result)

def reranked_search(index, query: str, namespace: str) -> None:
    """
    Performs a reranked search on the index using a reranker model.

    Args:
        index: Pinecone index object.
        query (str): Search query text.
        namespace (str): Namespace to search in.
    """
    info("🎯 Performing reranked search using BGE model")

    result = index.search(
        namespace=namespace,
        query={
            "top_k": 5,
            "inputs": {
                "text": query
            }
        },
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": 5,
            "rank_fields": ["chunk_text"]
        },
        fields=["category", "chunk_text"]
    )

    info("📄 Reranked Search Results:")
    info(result)