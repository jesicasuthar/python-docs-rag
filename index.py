import json
from pathlib import Path

import chromadb

# Configuration
INPUT_FILE = Path("data/processed/embedded_chunks.jsonl")
DB_PATH = "chroma_db"
COLLECTION_NAME = "python_documentation"
BATCH_SIZE = 250


def build_index():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run embed.py first."
        )

    # Create or connect to persistent local database
    client = chromadb.PersistentClient(path=DB_PATH)

    # Rebuild the collection to avoid stale or duplicate entries
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = []
    documents = []
    embeddings = []
    metadatas = []
    indexed_count = 0
    seen_ids = set()

    print("Indexing embedded chunks into ChromaDB...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_number}")
                continue

            chunk_id = record.get("id")
            text = record.get("text", "").strip()
            embedding = record.get("embedding")
            metadata = record.get("metadata") or {}

            # Validate required fields
            if not chunk_id or not text or not embedding:
                print(f"Skipping incomplete record on line {line_number}")
                continue

            chunk_id = str(chunk_id)

            if chunk_id in seen_ids:
                print(f"Skipping duplicate chunk ID: {chunk_id}")
                continue

            if len(embedding) != 384:
                print(
                    f"Skipping {chunk_id}: expected 384 dimensions, "
                    f"got {len(embedding)}"
                )
                continue

            # ChromaDB metadata supports primitive values
            clean_metadata = {}

            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[str(key)] = value
                elif value is not None:
                    clean_metadata[str(key)] = str(value)

            ids.append(chunk_id)
            documents.append(text)
            embeddings.append(embedding)
            metadatas.append(clean_metadata)
            seen_ids.add(chunk_id)

            # Write records in batches to avoid large memory usage
            if len(ids) >= BATCH_SIZE:
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

                indexed_count += len(ids)
                print(f"Indexed {indexed_count:,} chunks")

                ids = []
                documents = []
                embeddings = []
                metadatas = []

    # Insert remaining records
    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        indexed_count += len(ids)

    # Verify the collection
    stored_count = collection.count()

    print("\n========== CHROMADB INDEX REPORT ==========")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chunks indexed: {indexed_count:,}")
    print(f"Chunks stored: {stored_count:,}")
    print(f"Database path: {Path(DB_PATH).resolve()}")
    print("Embedding dimensions: 384")
    print("Text, embeddings, metadata, and IDs stored.")
    print("===========================================")

    if indexed_count != stored_count:
        print("WARNING: Indexed and stored counts do not match.")


if __name__ == "__main__":
    build_index()