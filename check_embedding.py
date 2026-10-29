import json
from pathlib import Path

path = Path("data/processed/embedded_chunks.jsonl")

with path.open("r", encoding="utf-8") as file:
    first_record = json.loads(next(line for line in file if line.strip()))

embedding = first_record["embedding"]

print("Chunk ID:", first_record["id"])
print("Source:", first_record["metadata"].get("source"))
print("Section:", first_record["metadata"].get("section"))
print("Embedding dimensions:", len(embedding))
print("First five values:", embedding[:5])

assert len(embedding) == 384
print("Embedding verification successful!")