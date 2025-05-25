"""
Main orchestration script: initialize client, prepare index, insert data, perform searches.
"""
import sys
import os

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from config import get_milvus_client, get_embedding_model, COLLECTION_NAME, VECTOR_DIM
from index_utils import recreate_collection, prepare_data, insert_data
from search_utils import search_vectors
from src.utils import info, error


def main():
    try:
        client = get_milvus_client()
        model = get_embedding_model()

        # Recreate collection
        recreate_collection(client, COLLECTION_NAME, VECTOR_DIM)

        # Initial chunk texts
        chunk_texts = [
            "The Eiffel Tower was completed in 1889 and stands in Paris, France.",
            "Photosynthesis allows plants to convert sunlight into energy.",
            "Albert Einstein developed the theory of relativity.",
            "The mitochondrion is often called the powerhouse of the cell.",
            "Shakespeare wrote many famous plays, including Hamlet and Macbeth.",
            "Water boils at 100°C under standard atmospheric pressure.",
            "The Great Wall of China was built to protect against invasions.",
            "Honey never spoils due to its low moisture content and acidity.",
            "The speed of light in a vacuum is approximately 299,792 km/s.",
            "Newton's laws describe the motion of objects."
        ]

        # Encode embeddings
        embeddings = model.encode(chunk_texts).tolist()
        data = prepare_data(chunk_texts, embeddings)
        info(f"Data ready with {len(data)} entities.")

        # Insert data
        insert_data(client, COLLECTION_NAME, data)

        # Search example
        query = ["Who is Alan Turing?"]
        query_vectors = model.encode(query).tolist()
        res = search_vectors(
            client,
            collection_name=COLLECTION_NAME,
            query_vectors=query_vectors,
            limit=1,
            output_fields=['id', 'text']
        )
        info(f"Search results for query '{query[0]}': {res}")

        # Insert more documents on a different subject
        docs = [
            "Machine learning has been used for drug design.",
            "Computational synthesis with AI algorithms predicts molecular properties.",
            "DDR1 is involved in cancers and fibrosis.",
        ]

        # Encode doc embeddings
        doc_embeddings = model.encode(docs).tolist()

        # Prepare doc data with 'subject' field
        doc_data = []
        start_id = len(data)
        for i, doc in enumerate(docs):
            doc_data.append({
                "id": start_id + i,
                "vector": doc_embeddings[i],
                "text": doc,
                "subject": "biology"
            })
        info(f"Prepared {len(doc_data)} new documents with subject 'biology'.")

        insert_data(client, COLLECTION_NAME, doc_data)

        # Search with filter example
        filtered_query = ["tell me AI related information"]
        filtered_vectors = model.encode(filtered_query).tolist()
        filtered_res = search_vectors(
            client,
            collection_name=COLLECTION_NAME,
            query_vectors=filtered_vectors,
            limit=2,
            output_fields=["text", "id"],
            filter_expr="subject == 'biology'"
        )
        info(f"Filtered search results: {filtered_res}")

    except Exception as exc:
        error(f"Exception in main: {exc}", service="main")
        raise


if __name__ == "__main__":
    main()
