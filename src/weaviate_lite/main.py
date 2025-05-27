"""
Main script to initialize Weaviate, create schema, insert documents,
and perform vector-based search using SentenceTransformers.
"""

import sys
import os

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from src.weaviate_lite.search_utils import search_documents
from src.weaviate_lite.config import get_weaviate_client, get_embedding_model, CLASS_NAME
from src.weaviate_lite.index_utils import create_schema, insert_documents

from src.utils import info

def main() -> None:
    """
    Entry point of the script. Initializes client and embedding model,
    creates schema, inserts documents, and performs a semantic search.
    """
    # Initialize Weaviate client and embedding model
    client = get_weaviate_client()
    model = get_embedding_model()

    # Create the schema
    create_schema(client, CLASS_NAME)

    # Documents and their embeddings
    texts = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI."
    ]
    embeddings = model.encode(texts).tolist()

    # Insert documents
    insert_documents(client, texts, embeddings, CLASS_NAME)

    # Perform a vector search
    query = "Who is Alan Turing?"
    query_vector = model.encode([query])[0].tolist()
    results = search_documents(client, query_vector, CLASS_NAME)

    # Display results
    hits = results.get("data", {}).get("Get", {}).get(CLASS_NAME, [])
    for result in hits:
        info(result)


if __name__ == "__main__":
    main()
