import json
import csv
from pathlib import Path

from rag import RAGAssistant, normalize_source_path

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "evaluation_dataset.json"
RESULTS_JSON_PATH = PROJECT_DIR / "data" / "processed" / "evaluation_results.json"
RESULTS_CSV_PATH = PROJECT_DIR / "data" / "processed" / "evaluation_results.csv"
EXPERIMENTS = [
    ("semantic_top10", {"diversify": False}),
    ("source_diversity_top10", {"diversify": True}),
    ("cross_encoder_top10", {"diversify": False, "rerank": True}),
    ("hybrid_top10", {"diversify": False, "hybrid": True}),
    (
        "hybrid_cross_encoder_top10",
        {"diversify": False, "hybrid": True, "rerank": True}
    ),
]


def source_matches(source, patterns):
    """Match expected paths against normalized source paths or their suffixes."""
    source = normalize_source_path(source).strip("/").casefold()

    return any(
        source == normalized_pattern
        or source.endswith(f"/{normalized_pattern}")
        for normalized_pattern in (
            normalize_source_path(pattern).strip("/").casefold()
            for pattern in patterns
        )
    )


def main():
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        test_cases = json.load(f)

    assistant = RAGAssistant(load_generation_model=False)
    experiment_results = []

    for experiment_name, options in EXPERIMENTS:
        results = []
        hits = 0
        print(
            f"\nEvaluating {len(test_cases)} questions for "
            f"{experiment_name} ...\n"
        )

        for case in test_cases:
            question = case["question"]
            expected = case["expected_source_patterns"]
            chunks = assistant.retrieve(
                question,
                top_k=10,
                **options
            ) or []

            retrieved_sources = [
                {
                    "id": chunk.get("id"),
                    "source": chunk.get("source"),
                    "source_path": chunk.get("source_path"),
                    "distance": chunk.get("distance"),
                }
                for chunk in chunks
            ]

            hit = any(
                source_matches(item["source"], expected)
                for item in retrieved_sources
            )
            hits += int(hit)

            row = {
                "id": case["id"],
                "question": question,
                "topic": case.get("topic", ""),
                "expected_source_patterns": expected,
                "retrieved_sources": retrieved_sources,
                "source_hit_at_10": hit,
            }

            results.append(row)
            status = "HIT " if row["source_hit_at_10"] else "MISS"
            retrieved = ", ".join(
                str(x["source"]) for x in row["retrieved_sources"]
            ) or "(no sources)"
            print(f'{row["id"]} [{status}] {question}')
            print(f'    expected: {", ".join(expected)}')
            print(f'    retrieved: {retrieved}')

        total = len(results)
        hit_rate = hits / total if total else 0.0
        experiment_results.append({
            "name": experiment_name,
            "options": options,
            "hits": hits,
            "total_questions": total,
            "source_hit_rate_at_10": hit_rate,
            "source_hit_rate_at_10_percent": round(hit_rate * 100, 2),
            "results": results,
        })
        print(
            f"\n{experiment_name}: {hits}/{total} "
            f"= {hit_rate:.3f} ({hit_rate * 100:.2f}%)"
        )

    baseline = experiment_results[0]
    summary = {
        "metric": "SourceHitRate@10",
        "definition": (
            "Fraction of questions where at least one of the top 10 retrieved "
            "chunks has a canonical source path matching an expected source "
            "pattern. Sphinx _sources/*.rst.txt paths are mapped to their "
            "corresponding rendered *.html paths."
        ),
        "k": 10,
        "hits": baseline["hits"],
        "total_questions": baseline["total_questions"],
        "source_hit_rate_at_10": baseline["source_hit_rate_at_10"],
        "source_hit_rate_at_10_percent": baseline[
            "source_hit_rate_at_10_percent"
        ],
        "results": baseline["results"],
        "experiments": experiment_results,
    }

    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "question", "topic", "expected_source_patterns",
                "experiment", "retrieved_sources", "source_hit_at_10"
            ],
        )
        writer.writeheader()
        for experiment in experiment_results:
            for row in experiment["results"]:
                writer.writerow({
                    **row,
                    "experiment": experiment["name"],
                    "expected_source_patterns": " | ".join(
                        row["expected_source_patterns"]
                    ),
                    "retrieved_sources": " | ".join(
                        f'{x["source"]} (indexed={x["source_path"]}, '
                        f'id={x["id"]}, distance={x["distance"]})'
                        for x in row["retrieved_sources"]
                    ),
                })

    print("\n" + "=" * 56)
    for experiment in experiment_results:
        print(
            f'{experiment["name"]}: {experiment["hits"]}/'
            f'{experiment["total_questions"]} = '
            f'{experiment["source_hit_rate_at_10"]:.3f} '
            f'({experiment["source_hit_rate_at_10_percent"]:.2f}%)'
        )
    print(f"JSON results: {RESULTS_JSON_PATH}")
    print(f"CSV results:  {RESULTS_CSV_PATH}")
    print("=" * 56)
    print(
        "\nNote: This is a lightweight source-path match evaluation. "
        "Review the expected patterns and results manually before treating "
        "the score as a benchmark."
    )


if __name__ == "__main__":
    main()
