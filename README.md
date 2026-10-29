# Python Documentation Assistant

---
title: Python Documentation Assistant
emoji: 🐍
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
python_version: "3.11"
startup_duration_timeout: 1h
---

Ask beginner-friendly Python questions and receive answers grounded in
retrieved Python 3.12 documentation, with source labels and expandable
retrieved passages.

## Live demo

[Open the Hugging Face Space](https://huggingface.co/spaces/jesica2004/python-docs-rag)

**Deployment status:** The Space URL exists, but at the last check it still
showed Hugging Face's default static Space page rather than this application.
The RAG app is not yet available at the demo link.

## Architecture

```text
Question
   |
   v
SentenceTransformer: all-MiniLM-L6-v2
   |
   v
ChromaDB: python_documentation collection
   |  top-3 chunks; source diversity enabled
   v
Retrieved excerpts + source labels
   |
   v
Qwen2.5-0.5B-Instruct (local CPU generation)
   |
   v
Grounded answer, citations, sources, and expandable context
```

- **Corpus:** Python 3.12 documentation, indexed as 28,645 chunks.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector store:** persistent local ChromaDB collection named
  `python_documentation`.
- **Answer model:** `Qwen/Qwen2.5-0.5B-Instruct`, loaded locally by
  Transformers and run on CPU.
- **Interface:** Streamlit. The assistant is cached for the app process, and
  interactions are appended to `data/processed/interactions.jsonl`.
- **Retrieval:** the UI requests three results. The retrieval function uses a
  larger candidate pool and source diversity by default to limit repeated
  chunks from the same documentation page.

## Run locally

Use Python 3.11 and a copy of the project source. The ChromaDB index is not
included in the source repository; place the existing index in `chroma_db/`
or set `CHROMA_DB_PATH` to its location.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:CHROMA_DB_PATH = "C:\path\to\rag-project\chroma_db"
streamlit run app.py
```

On macOS or Linux, create and activate a virtual environment, install
`requirements.txt`, and run:

```bash
export CHROMA_DB_PATH="/path/to/rag-project/chroma_db"
streamlit run app.py
```

The first run downloads the embedding model and language model. Without
`CHROMA_DB_PATH`, the app looks for a `chroma_db/` directory next to `rag.py`.
That directory must contain `chroma.sqlite3` and the associated Chroma index
files for the `python_documentation` collection.

### Optional Hugging Face index download

The app can download the existing ChromaDB snapshot from a Hugging Face Dataset
repository if the local index is absent:

```powershell
$env:CHROMA_DATASET_ID = "jesica2004/python-docs-rag-index"
# Set HF_TOKEN only if the dataset is private.
streamlit run app.py
```

The Dataset repository must contain the **contents** of `chroma_db/` at its
root, including `chroma.sqlite3` and the UUID-named index directories. The
Dataset is not included in the application source. The configured Dataset
repository is currently empty, so the index must be uploaded before a hosted
app can start.

## Deploy on Streamlit Community Cloud

Community Cloud deploys from GitHub. The local `chroma_db/` directory is about
522 MB and must stay out of the application repository; upload its **contents**
to the Hugging Face Dataset repository described above first. The dataset must
be accessible to the app at startup. For a private dataset, create a
Hugging Face read token.

From the project directory, authenticate interactively with a Hugging Face
account that has write access to the Dataset, then upload the database
directory. The CLI uploads the directory contents to the Dataset root and can
resume an interrupted upload:

```powershell
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\hf.exe upload jesica2004/python-docs-rag-index --repo-type dataset .\chroma_db
```

1. Create a GitHub repository and push the application source. Keep `.venv/`,
   `__pycache__/`, `logs/`, `chroma_db/`, the rebuild-only `data/raw/` corpus,
   and generated chunk/evaluation output files out of Git; `.gitignore`
   excludes these.
2. Confirm that the Hugging Face Dataset contains `chroma.sqlite3` and all
   UUID-named Chroma index directories at its root.
3. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create
   app**, then select the GitHub repository, branch, and `app.py` entrypoint.
   In **Advanced settings**, choose Python 3.11 and add these secrets:

   ```toml
   CHROMA_DATASET_ID = "jesica2004/python-docs-rag-index"
   # Add only if the dataset is private:
   # HF_TOKEN = "your-read-token"
   ```

4. Deploy and check the app logs if startup fails. On first startup, the app
   downloads the Chroma index and transformer models, so loading can take time.
   The downloaded index is local to the running instance and may need to be
   downloaded again after a restart.

The existing `Dockerfile` is for Docker-based hosting and is not needed by
Community Cloud; the app's Python packages are installed from
`requirements.txt`.

## Example questions

- How do I open and read a CSV file in Python?
- What is the difference between `read()`, `readline()`, and `readlines()`?
- How do I define a function with a default parameter value?
- How do list comprehensions work?
- How do I sort a list in place versus returning a sorted copy?
- How can I create a virtual environment for a Python project?
- How do I parse command-line arguments in a Python script?
- How do I serialize and deserialize JSON data in Python?

## Known limitations

- **The public demo is not deployed yet.** The current Hugging Face Space URL
  still displays the platform's default static page.
- **The index is external and large.** The local ChromaDB directory is about
  547 MB and is gitignored. A hosted deployment needs the index uploaded
  separately and configured through `CHROMA_DATASET_ID` or a mounted volume.
- **Cloud storage is ephemeral by default.** A Space may need to download the
  index again after a restart unless persistent external storage is configured.
- **First-run downloads and CPU generation can be slow.** Both transformer
  models are fetched at runtime, and the language model runs on CPU.
- **Answers can be wrong or incomplete.** The grounding prompt and citation
  labels do not guarantee factual correctness. Check the retrieved excerpts
  and documentation for important decisions.
- **Evaluation is small and source-oriented.** The benchmark has 20 questions
  with weak expected-source labels. Its `SourceHitRate@k` measures whether an
  expected documentation page appears in the first `k` results; it does not
  measure answer correctness. In current local runs, semantic `SourceHitRate@3`
  was 12/20 (60%), while source-diverse `SourceHitRate@10` was 20/20 (100%).
- **Manual answer review is incomplete.** The review sheet has generated
  answers for 20 questions, but no human classifications have been recorded.
- **Interaction logs are local files.** Logs are written to
  `data/processed/interactions.jsonl`; on ephemeral hosts they may be lost on
  restart. Do not enter sensitive information.

For benchmark details, methodology, results, and improvement ideas, see
[EVALUATION.md](./EVALUATION.md).
