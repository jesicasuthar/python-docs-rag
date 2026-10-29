import chromadb

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="python_documentation"
)

print("Total chunks:", collection.count())

sample = collection.get(
    limit=1,
    include=["documents", "metadatas", "embeddings"]
)

if sample["ids"]:
    print("\nChunk ID:")
    print(sample["ids"][0])

    print("\nText preview:")
    print(sample["documents"][0][:300])

    print("\nMetadata:")
    print(sample["metadatas"][0])

    print("\nEmbedding dimensions:")
    print(len(sample["embeddings"][0]))
else:
    print("The collection is empty.")