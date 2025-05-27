"""
Schema definition and data insertion for Weaviate.

This module defines the vector schema and handles inserting text data
with corresponding embeddings into a Weaviate vector database.
"""

import sys
import os
from typing import List

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from src.utils import info, error


def create_schema(client, class_name: str = "Document") -> None:
    """
    Creates a Weaviate schema with the specified class name.

    The schema includes one property: 'text' of type 'text', and no built-in vectorizer.

    Args:
        client (weaviate.Client): The initialized Weaviate client.
        class_name (str): The name of the class to create. Default is "Document".

    Raises:
        Exception: If the schema creation fails.
    """
    schema = {
        "classes": [{
            "class": class_name,
            "vectorizer": "none",
            "properties": [{
                "name": "text",
                "dataType": ["text"]
            }]
        }]
    }
    try:
        client.schema.delete_all()
        client.schema.create(schema)
        info("📦 Schema created for class '{}'".format(class_name))
    except Exception as exc:
        error("❌ Failed to create schema: {}".format(str(exc)))
        raise


def insert_documents(
    client,
    texts: List[str],
    vectors: List[List[float]],
    class_name: str = "Document"
) -> None:
    """
    Inserts documents with their corresponding vectors into Weaviate.

    Args:
        client (weaviate.Client): The initialized Weaviate client.
        texts (List[str]): List of document texts.
        vectors (List[List[float]]): Corresponding list of embedding vectors.
        class_name (str): The class name into which the documents will be inserted.

    Raises:
        Exception: If the insertion of any document fails.
    """
    try:
        for _, (text, vector) in enumerate(zip(texts, vectors)):
            client.data_object.create(
                data_object={"text": text},
                class_name=class_name,
                vector=vector
            )
        info("✅ Inserted {} documents into '{}'".format(len(texts), class_name))
    except Exception as exc:
        error("❌ Failed to insert documents: {}".format(str(exc)))
        raise
