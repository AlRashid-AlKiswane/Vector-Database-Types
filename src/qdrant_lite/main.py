import sys
import os

# Fix import path for development
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.qdrant_lite.config import get_qdrant_client, get_embedding_model, COLLECTION_NAME, VECTOR_DIM
from src.qdrant_lite.index_utils import create_qdrant_collection, insert_data
from src.qdrant_lite.search_utils import search_qdrant

def main():
    client = get_qdrant_client()
    model = get_embedding_model()

    create_qdrant_collection(client, COLLECTION_NAME, VECTOR_DIM)

    texts = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI."
    ]
    embeddings = model.encode(texts).tolist()
    insert_data(client, COLLECTION_NAME, embeddings, texts)

    query = model.encode(["Who is Alan Turing?"]).tolist()[0]
    results = search_qdrant(client, COLLECTION_NAME, query)

    for result in results:
        print(result)

if __name__ == "__main__":
    main()
