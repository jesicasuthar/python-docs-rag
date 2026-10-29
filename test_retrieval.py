import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "chroma_db"
COLLECTION_NAME = "python_documentation"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def test_retrieval():
    # Connect to the persistent ChromaDB database
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Indexed chunks: {collection.count():,}")

    if collection.count() == 0:
        raise RuntimeError("The ChromaDB collection is empty.")

    # Load the same embedding model used during ingestion
    model = SentenceTransformer(MODEL_NAME)

    # Test query
    query = "How can I read a CSV file in Python?"

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Retrieve the five closest matching chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    if not documents:
        print("No matching chunks found.")
        return

    print(f"\nQuery: {query}")
    print(f"Results returned: {len(documents)}")

    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1
    ):
        print(f"\n{'=' * 60}")
        print(f"Result {rank}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Source: {metadata.get('source', 'Not specified')}")
        print(f"Cosine distance: {distance:.4f}")
        print("\nText:")
        print(document[:600])

    print("\nRetrieval test completed.")


if __name__ == "__main__":
    test_retrieval()