"""
Main script for interacting with Pinecone: index creation, upserting records,
searching, and reranking using a pretrained embedding model.
"""
import sys
import os

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from config import PINECONE_API_KEY, INDEX_NAME
from index_utils import init_pinecone, create_index_if_needed, upsert_sample_records
from search_utils import basic_search, reranked_search
from src.utils import error, info



def main():
    """
    Main function to run Pinecone index operations:
    - Initialize Pinecone
    - Create index (if not exists)
    - Upsert records
    - Perform search and reranked search
    """
    try:
        info("🚀 Starting Pinecone operations pipeline")

        # Step 1: Initialize Pinecone client
        pc = init_pinecone(api_key=PINECONE_API_KEY)

        # Step 2: Create index if it doesn't exist
        create_index_if_needed(pc, INDEX_NAME)

        # Step 3: Upsert data to index
        index = pc.Index(INDEX_NAME)
        upsert_sample_records(index, namespace="ns1")

        # Step 4: Basic search
        query = "historical structures and monuments"
        basic_search(index, query=query, namespace="ns1")

        # Step 5: Reranked search
        reranked_search(index, query=query, namespace="ns1")

    except (KeyError, ValueError, RuntimeError) as e:
        error("❌ An error occurred: %s", e)

if __name__ == "__main__":
    main()
