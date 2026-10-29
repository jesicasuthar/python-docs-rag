import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configuration
INPUT_FILE = Path("data/processed/chunks.jsonl")
OUTPUT_FILE = Path("data/processed/embedded_chunks.jsonl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


def generate_embeddings():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {INPUT_FILE}. "
            "Run ingest.py first."
        )

    # Load the processed chunks
    chunks = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON at line {line_number}: {exc}")
                continue

            text = record.get("text", "").strip()

            if not text:
                continue

            chunks.append(record)

    if not chunks:
        print("No valid chunks found.")
        return

    print(f"Loaded {len(chunks)} chunks.")

    # Load the local embedding model
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings in batches
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save chunks and embeddings as JSONL
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk, embedding in zip(chunks, embeddings):
            output_record = {
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "embedding": embedding.tolist()
            }

            file.write(
                json.dumps(output_record, ensure_ascii=False) + "\n"
            )

    print("\n========== EMBEDDING REPORT ==========")
    print(f"Model: {MODEL_NAME}")
    print(f"Chunks embedded: {len(embeddings)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Output file: {OUTPUT_FILE.resolve()}")
    print("======================================")


if __name__ == "__main__":
    generate_embeddings()