import csv
import json
from pathlib import Path
from rag import RAGAssistant

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "evaluation_dataset.json"
OUTPUT = ROOT / "data" / "processed" / "manual_answer_review.csv"

CLASSIFICATIONS = {
    "Accurate",
    "Partially Accurate",
    "Incorrect",
    "Not Answerable",
}


def main():
    with DATASET.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    assistant = RAGAssistant()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"Generating answers for {len(cases)} review questions...")

    for i, case in enumerate(cases, start=1):
        question = case["question"]
        print(f"[{i}/{len(cases)}] {case['id']}: {question}")

        try:
            chunks = assistant.retrieve(question, top_k=3) or []
            answer = (
                assistant.generate_answer(question, chunks)
                if chunks
                else ""
            )
            sources = " | ".join(
                f'{chunk.get("source", "")} [id={chunk.get("id", "")}; '
                f'distance={chunk.get("distance", "")}]'
                for chunk in chunks
            )
            error = ""
        except Exception as exc:
            chunks = []
            answer = ""
            sources = ""
            error = f"{type(exc).__name__}: {exc}"

        rows.append({
            "id": case["id"],
            "question": question,
            "expected_source_patterns": " | ".join(
                case.get("expected_source_patterns", [])
            ),
            "retrieved_sources": sources,
            "generated_answer": answer,
            "classification": "",
            "review_notes": error,
        })

    fieldnames = [
        "id", "question", "expected_source_patterns",
        "retrieved_sources", "generated_answer",
        "classification", "review_notes"
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCreated review sheet: {OUTPUT}")
    print("Open it in Excel and fill classification with exactly one of:")
    for label in sorted(CLASSIFICATIONS):
        print(f" - {label}")
    print("Use review_notes to explain omissions, unsupported claims, or errors.")


if __name__ == "__main__":
    main()
