RAG Evaluation Report
1. Project Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot for
Python programming questions. It retrieves passages from the Python
documentation using semantic search and uses a locally hosted language
model to generate answers grounded in the retrieved context.

Main components - Embedding model:
sentence-transformers/all-MiniLM-L6-v2 - Vector store: ChromaDB -
LLM: Qwen/Qwen2.5-0.5B-Instruct - Interface: Streamlit -
Indexed corpus: Python 3.12 documentation
2. Dataset and Index Details
  Item                   Details
  Documentation corpus   Python 3.12 documentation
  Indexed files          1,036
  Text chunks            28,645
  Evaluation questions   20
  Retrieval metric       SourceHitRate@3
  Retrieval setting      Up to 3 retrieved chunks per question

The evaluation dataset is stored in evaluation_dataset.json. It
contains 20 Python documentation questions and expected source-path
patterns for checking whether retrieval returned a relevant
documentation source.

Topics include CSV reading/writing, file methods and exceptions,
exception handling, function arguments, list comprehensions, sets,
sorting, dictionaries, imports, virtual environments, pip, classes,
enumerate(), pathlib, argparse, and JSON.

This is a small, manually assembled evaluation set. It provides an
initial diagnostic signal, not a comprehensive benchmark of Python
knowledge or all possible user questions.
3. Retrieval Results
SourceHitRate@3
Initial result: 10 / 20 = 50.00%

SourceHitRate@3 measures the proportion of evaluation questions for
which at least one of the top three retrieved chunks matches an expected
source pattern.

SourceHitRate@3 = questions with a matching expected source in the top 3 / total evaluation questions

For this run: 10 / 20 = 0.50 = 50.00%.

Questions recorded as hits - Q01 --- CSV reading - Q02 --- CSV
writing - Q04 --- read(), readline(), and readlines() - Q05 ---
missing-file exceptions - Q10 --- sets - Q13 --- imports - Q14 ---
virtual environments - Q15 --- pip - Q18 --- pathlib - Q19 ---
argparse

Questions recorded as misses - Q03 --- opening files for reading
with open() - Q06 --- try / except / else / finally - Q07 ---
default arguments - Q08 --- *args and **kwargs - Q09 --- list
comprehensions - Q11 --- in-place list.sort() versus sorted() - Q12
--- dictionaries - Q16 --- classes and __init__ - Q17 ---
enumerate() - Q20 --- JSON
Interpretation and caveats
The initial recorded SourceHitRate@3 is 50%, suggesting that
retrieval and/or evaluation matching needs improvement. However, a
recorded miss does not always prove that retrieval failed. Some
expected-source patterns may not match the way indexed documentation
paths are represented (for example, an HTML page versus a corresponding
_sources/*.rst.txt file). Inspect the retrieved source paths and
validate the expected patterns before treating every miss as a true
retrieval failure.

The target of 90% would mean at least 18 of the 20 questions
register as hits under the current metric. This is a target, not a
result or guarantee.
4. Answer Quality Results
Status: Manual answer-quality review has not yet been completed.

The retrieval score measures whether expected source paths appear in the
top three results. It does not measure whether generated answers are
correct, complete, understandable, or properly supported by the
retrieved context.

Use data/processed/manual_answer_review.csv to review generated
answers. For each test question, inspect the answer and retrieved
evidence, then record the review outcome using the labels and
instructions in MANUAL_REVIEW_GUIDE.txt (if present).

Recommended answer-quality dimensions:
  Dimension                           What to check
  Correctness                         Does the answer accurately answer
                                      the question?

  Grounding                           Are its claims supported by the
                                      retrieved documentation?

  Completeness                        Does it cover the important parts
                                      of the question?

  Relevance                           Does it avoid unrelated
                                      information?

  Source usefulness                   Do the displayed source references
                                      help users verify the answer?

  Abstention                          Does it acknowledge insufficient
                                      evidence instead of inventing an
                                  answer?
Until this review is completed, no numeric answer-quality score should
be reported. The 50% SourceHitRate@3 must not be presented as an
answer-correctness score.
5. Improvement Ideas
A. Validate the evaluation before changing the RAG pipeline
1. Review expected source patterns in evaluation_dataset.json.
2. Compare them with the actual source paths returned for each
   question.
3. Normalize equivalent documentation paths, such as HTML pages and
   indexed source-text equivalents, only when they genuinely refer to
   the same topic.
4. Re-run the evaluation and document changes to the matching rules. Do
   not broaden matching so much that unrelated sources count as hits.
B. Diagnose retrieval misses
For every miss, inspect the question, top-three retrieved chunks, source
paths, and distance values. Determine whether the cause is: - An
incorrect or overly strict expected-source pattern - Relevant
documentation missing from the index - Poor chunk boundaries or
insufficient surrounding context - A query/document wording mismatch -
Ranking that places relevant material below the top three results

ChromaDB distance is a distance value, not a similarity percentage;
lower values generally indicate closer matches for the configured
distance metric.
C. Improve retrieval quality
- Experiment with chunk size and overlap so definitions and examples
  remain together.
- Preserve useful headings and section context during ingestion.
- Consider retrieving more candidates initially and reranking them
  before selecting the final context.
- Compare the current embedding model with another suitable local
  embedding model using the same fixed evaluation set.
- Tune top_k based on measured quality and latency, rather than
  increasing it solely to improve the score.
- Rebuild the index whenever the corpus, chunking strategy, or
  embedding model changes.
D. Improve answer generation
- Instruct the model to answer from the supplied documentation context
  and not invent unsupported details.
- Ask it to say when the context does not contain enough information.
- Keep retrieved context focused and remove duplicated or irrelevant
  chunks.
- Include clear source references in the UI.
- Test whether the small local model follows grounding instructions
  reliably; record errors instead of assuming the prompt alone will
  solve them.
E. Complete a manual answer review
- Generate or refresh data/processed/manual_answer_review.csv.
- Review all 20 answers against their retrieved context and the
  relevant Python documentation.
- Record correctness, grounding, completeness, and any hallucinations
  or omissions.
- Summarize the number reviewed, number passing, common failure types,
  and resulting answer-quality percentage.
- Re-test after changes using the same questions so results remain
  comparable.
F. Expand and maintain the evaluation set
- Add questions for common beginner, intermediate, and troubleshooting
  tasks.
- Include questions that require combining multiple documentation
  sections.
- Include unanswerable or underspecified questions to test safe
  abstention.
- Keep expected-source patterns specific and review them when
  documentation paths change.
- Track retrieval quality, answer quality, and response latency as
  separate measures.
6. Reproducibility and Reporting
For each evaluation run, record: - Date and time - Dataset version -
Corpus and index version - Embedding model - Chunk size and overlap -
Retrieval top_k - SourceHitRate@3 - Manual answer-review results, when
available - Known limitations or changes to the evaluation matcher

Avoid comparing scores from runs that use different datasets or
materially different matching rules without clearly documenting those
changes.
7. Current Summary
  Measure                             Current status
  Indexed documentation files         1,036

  Indexed chunks                      28,645

  Evaluation questions                20

  Initial SourceHitRate@3             10/20 (50.00%)

  Target SourceHitRate@3              18/20 (90.00%)

  Manual answer-quality score         Not yet measured

  Primary next step                   Validate source-path matching,
                                      inspect misses, and complete manual
                                  answer review
Conclusion: The current evaluation provides a baseline, not proof of
overall chatbot quality. First verify that expected-source matching is
fair, then diagnose genuine retrieval failures and manually assess
generated answers. Report improvements only after rerunning the
evaluation and recording the evidence.