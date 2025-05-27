"""
Weaviate configuration and embedding model loading module.

This module provides functions to initialize a Weaviate client and load
a SentenceTransformer model for vector embeddings.

Attributes:
    ROOT_DIR (str): Absolute path to the root directory of the project.
    CLASS_NAME (str): Default class name used in the Weaviate schema.
"""

import os
import sys

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

import weaviate
from weaviate.exceptions import WeaviateBaseError
from weaviate.classes.init import AdditionalConfig, Timeout
from sentence_transformers import SentenceTransformer
from src.utils import info, error
from src.utils import info, error  # Assuming your logger is in this path

CLASS_NAME = "Document"


def get_weaviate_client() -> weaviate.WeaviateClient:
    """
    Initialize and return a Weaviate client.

    Connects to a local Weaviate instance running on http://localhost:8080.
    Includes support for gRPC port and timeout configuration.

    Returns:
        weaviate.WeaviateClient: An initialized Weaviate client instance.

    Raises:
        WeaviateBaseError: If the Weaviate client fails to connect properly.
        Exception: For any unexpected error.
    """
    try:
        info("🔄 Attempting to connect to Weaviate at http://localhost:8080...")

        client = weaviate.connect_to_local(
            port=8080,
            grpc_port=50051,
            headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY", "")  # Optional
            },
            additional_config=AdditionalConfig(
                timeout=Timeout(init=10, query=30, insert=60)
            )
        )

        info("✅ Weaviate client initialized successfully.")
        return client

    except WeaviateBaseError as exc:
        error(f"❌ WeaviateBaseError: Failed to connect to Weaviate - {exc}")
        raise
    except Exception as exc:
        error(f"❌ Unexpected error during Weaviate connection: {exc}")
        raise


def get_embedding_model() -> SentenceTransformer:
    """
    Load and return a SentenceTransformer embedding model.

    Loads the 'all-MiniLM-L6-v2' model from SentenceTransformers.
    If loading fails, logs the error and raises the original exception.

    Returns:
        SentenceTransformer: A loaded SentenceTransformer model.

    Raises:
        Exception: If the model fails to load for any reason.
    """
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        info("📐 Embedding model 'all-MiniLM-L6-v2' loaded successfully.")
        return model
    except Exception as exc:
        error("❌ Failed to load embedding model: {}".format(str(exc)))
        raise
    finally:
        info("ℹ️ Embedding model loading process complete.")
