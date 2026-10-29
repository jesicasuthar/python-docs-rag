# RAG retrieval evaluation

Files:
- `evaluation_dataset.json`: 20 Python documentation questions and expected source-path patterns.
- `evaluate_retrieval.py`: runs five retrieval experiments over all 20 questions using `top_k=10`.
- `rag.py`: provides semantic retrieval, source diversity, hybrid retrieval, and optional cross-encoder reranking.
- `generate_manual_review.py`: optionally creates a spreadsheet for human review of generated answers.

## Install / run
Copy both files into your `rag-project` root, alongside `rag.py` and `app.py`.
With your virtual environment activated, run:

```powershell
python evaluate_retrieval.py
```

The script writes:
- `data/processed/evaluation_results.json`
- `data/processed/evaluation_results.csv`

By default, `rag.py` enables source diversity to avoid returning many chunks
from the same documentation page. The experiments compare semantic top-10
retrieval without diversity, source diversity, a cross-encoder reranker, hybrid
keyword + semantic retrieval, and hybrid retrieval with reranking. The reranker
model is loaded only for reranking experiments. Retrieval evaluation does not
load the answer-generation model.

## Metric definition
For each question, a hit is counted if at least one of the top 10 retrieved
chunks has a canonical source path containing one of that question's
`expected_source_patterns`. Indexed Sphinx `_sources/*.rst.txt` paths are
canonicalized to their corresponding rendered `*.html` paths for matching;
expected paths are compared as exact paths or path suffixes (not arbitrary
substrings), and the original indexed path is retained in results for
traceability.

`SourceHitRate@10 = number of hit questions / total evaluated questions`

This evaluates retrieval source coverage, not answer correctness. The expected source patterns are an initial weak-label set based on the standard Python documentation structure; check that they match the documentation version and file layout in your index. A zero-hit result can mean either retrieval failed or a source path/pattern differs from the indexed corpus.

## Optional answer review
Run `python generate_manual_review.py` to generate
`data/processed/manual_answer_review.csv` for human assessment of answer
accuracy, omissions, and unsupported claims. This is a separate, optional
answer-quality review; it is not part of the retrieval source-hit metric.