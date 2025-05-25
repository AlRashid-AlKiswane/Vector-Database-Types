"""
Index utilities for Milvus: collection management and data insertion.
"""
import sys
import os

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from pymilvus import MilvusClient
from src.utils import info, debug, warning, error
from typing import List, Dict, Any


def recreate_collection(client: MilvusClient, collection_name: str, dimension: int) -> None:
    """
    Drop collection if exists, then create a new one.

    Args:
        client (MilvusClient): Milvus client.
        collection_name (str): Collection name.
        dimension (int): Vector dimension.
    """
    try:
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            info(f"Dropped existing collection '{collection_name}'", service="index_utils")
        client.create_collection(collection_name=collection_name, dimension=dimension)
        info(f"Created collection '{collection_name}' with dimension {dimension}", service="index_utils")
    except Exception as exc:
        error(f"Failed to recreate collection '{collection_name}': {exc}", service="index_utils")
        raise


def prepare_data(chunk_texts: List[str], embeddings: List[List[float]]) -> List[Dict[str, Any]]:
    """
    Prepare data entries with id, vector, and text fields.

    Args:
        chunk_texts (List[str]): List of chunk texts.
        embeddings (List[List[float]]): Corresponding embeddings.

    Returns:
        List[Dict[str, Any]]: List of dicts suitable for Milvus insert.
    """
    data = []
    try:
        for i, text in enumerate(chunk_texts):
            data.append({
                "id": i,
                "vector": embeddings[i],
                "text": text,
            })
        info(f"Prepared {len(data)} data entries for insertion", service="index_utils")
        return data
    except Exception as exc:
        error(f"Error preparing data: {exc}", service="index_utils")
        raise


def insert_data(client: MilvusClient, collection_name: str, data: List[Dict[str, Any]]) -> None:
    """
    Insert data into Milvus collection.

    Args:
        client (MilvusClient): Milvus client.
        collection_name (str): Collection to insert into.
        data (List[Dict[str, Any]]): Data to insert.
    """
    try:
        client.insert(collection_name=collection_name, data=data)
        info(f"Inserted {len(data)} entities into collection '{collection_name}'", service="index_utils")
    except Exception as exc:
        error(f"Failed to insert data into collection '{collection_name}': {exc}", service="index_utils")
        raise
