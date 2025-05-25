"""
Utility functions for initializing Pinecone and managing index and data operations.
"""
import sys
import os

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
from pinecone import Pinecone
from src.utils import info


def init_pinecone(api_key: str) -> Pinecone:
    """
    Initializes the Pinecone client.

    Args:
        api_key (str): The API key for Pinecone.

    Returns:
        Pinecone: Initialized Pinecone client.
    """
    info("🔐 Initializing Pinecone client", service="Pinecone")
    return Pinecone(api_key=api_key)


def create_index_if_needed(pc: Pinecone, index_name: str) -> None:
    """
    Creates an index if it doesn't already exist.

    Args:
        pc (Pinecone): Pinecone client.
        index_name (str): Name of the index.
    """
    if not pc.has_index(index_name):
        info(f"📦 Creating index: {index_name}", service="Pinecone")
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            }
        )
    else:
        info(f"✅ Index already exists: {index_name}", service="Pinecone")


def upsert_sample_records(index, namespace: str) -> None:
    """
    Upserts sample educational records to the index.

    Args:
        index: Pinecone index object.
        namespace (str): Namespace to insert records into.
    """
    info(f"📝 Upserting sample records to namespace '{namespace}'", service="Pinecone")

    records = [
        {"_id": "rec1", "chunk_text": "The Eiffel Tower was completed in 1889 and stands in Paris, France.", "category": "history"},
        {"_id": "rec2", "chunk_text": "Photosynthesis allows plants to convert sunlight into energy.", "category": "science"},
        {"_id": "rec3", "chunk_text": "Albert Einstein developed the theory of relativity.", "category": "science"},
        {"_id": "rec4", "chunk_text": "The mitochondrion is often called the powerhouse of the cell.", "category": "biology"},
        {"_id": "rec5", "chunk_text": "Shakespeare wrote many famous plays, including Hamlet and Macbeth.", "category": "literature"},
        {"_id": "rec6", "chunk_text": "Water boils at 100°C under standard atmospheric pressure.", "category": "physics"},
        {"_id": "rec7", "chunk_text": "The Great Wall of China was built to protect against invasions.", "category": "history"},
        {"_id": "rec8", "chunk_text": "Honey never spoils due to its low moisture content and acidity.", "category": "food science"},
        {"_id": "rec9", "chunk_text": "The speed of light in a vacuum is approximately 299,792 km/s.", "category": "physics"},
        {"_id": "rec10", "chunk_text": "Newton's laws describe the motion of objects.", "category": "physics"}
    ]

    index.upsert_records(namespace, records)
    info("✅ Records successfully upserted.", service="Pinecone")
