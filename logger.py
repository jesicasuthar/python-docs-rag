import json
from datetime import datetime, timezone
from pathlib import Path


LOG_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "processed"
    / "interactions.jsonl"
)


def log_interaction(question, chunks, answer):
    """Append one RAG interaction to a JSONL file."""

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    interaction = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "retrieved_sources": [
            {
                "source_id": chunk.get("id"),
                "source": chunk.get("source"),
                "score": chunk.get("distance"),
            }
            for chunk in chunks
        ],
        "generated_answer": answer,
    }

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(interaction, ensure_ascii=False) + "\n"
        )