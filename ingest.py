from pathlib import Path
from bs4 import BeautifulSoup
import markdown
import json
import re
import hashlib
import statistics


# ---------------- CONFIGURATION ----------------

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"

# Chunking configuration
CHUNK_SIZE = 1200       # Approximate characters per chunk
CHUNK_OVERLAP = 150     # Overlap between fixed-size chunks
MIN_CHUNK_SIZE = 100    # Minimum characters to retain a chunk

SUPPORTED_EXTENSIONS = {".html", ".htm", ".md", ".txt"}


# ---------------- TEXT EXTRACTION ----------------

def extract_text(file_path):
    """Extract clean plain text from HTML, Markdown, or TXT."""

    try:
        content = file_path.read_text(
            encoding="utf-8",
            errors="replace"
        )
    except OSError as exc:
        print(f"Could not read {file_path}: {exc}")
        return "", []

    if not content.strip():
        return "", []

    # HTML documents
    if file_path.suffix.lower() in {".html", ".htm"}:
        soup = BeautifulSoup(content, "html.parser")

        # Remove elements that are not useful as document content
        for element in soup.select(
            "script, style, nav, footer, "
            "header, aside, noscript"
        ):
            element.decompose()

        # Prefer the main article content when available
        main = soup.find("main") or soup.find("article") or soup.body

        if main is None:
            main = soup

        # Extract sections using headings
        sections = []
        current_heading = "Introduction"
        current_lines = []

        for element in main.find_all(
            ["h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote"]
        ):
            text = element.get_text(" ", strip=True)

            if not text:
                continue

            if element.name in {"h1", "h2", "h3", "h4"}:
                if current_lines:
                    sections.append({
                        "section": current_heading,
                        "text": "\n".join(current_lines)
                    })

                current_heading = text
                current_lines = []
            else:
                current_lines.append(text)

        if current_lines:
            sections.append({
                "section": current_heading,
                "text": "\n".join(current_lines)
            })

        # Fallback if headings were not found
        if not sections:
            plain_text = main.get_text("\n", strip=True)
            return clean_text(plain_text), []

        return "", [
            {
                "section": item["section"],
                "text": clean_text(item["text"])
            }
            for item in sections
            if clean_text(item["text"])
        ]

    # Markdown documents
    if file_path.suffix.lower() == ".md":
        # Convert Markdown to HTML to identify headings
        html = markdown.markdown(
            content,
            extensions=["fenced_code"]
        )
        soup = BeautifulSoup(html, "html.parser")

        sections = []
        current_heading = "Introduction"
        current_lines = []

        for element in soup.find_all(
            ["h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote"]
        ):
            text = element.get_text(" ", strip=True)

            if not text:
                continue

            if element.name in {"h1", "h2", "h3", "h4"}:
                if current_lines:
                    sections.append({
                        "section": current_heading,
                        "text": "\n".join(current_lines)
                    })

                current_heading = text
                current_lines = []
            else:
                current_lines.append(text)

        if current_lines:
            sections.append({
                "section": current_heading,
                "text": "\n".join(current_lines)
            })

        sections = [
            {
                "section": item["section"],
                "text": clean_text(item["text"])
            }
            for item in sections
            if clean_text(item["text"])
        ]

        if sections:
            return "", sections

        # Fallback if no usable Markdown sections were found
        return clean_text(soup.get_text("\n", strip=True)), []

    # Plain text documents
    return clean_text(content), []


# ---------------- TEXT CLEANING ----------------

def clean_text(text):
    """Normalize whitespace and remove empty lines."""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# ---------------- CHUNKING ----------------

def fixed_size_chunks(text):
    """Split long text into overlapping character-based chunks."""

    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Prefer ending at a word boundary
        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - CHUNK_OVERLAP

        # Guarantee forward progress
        start = max(start + 1, next_start)

    return chunks


def split_into_chunks(text, sections):
    """Use header-based splitting with fixed-size fallback."""

    chunks = []

    if sections:
        for section in sections:
            section_text = section["text"]

            # Short sections are kept together.
            # Long sections use fixed-size splitting.
            if len(section_text) <= CHUNK_SIZE:
                section_chunks = [section_text]
            else:
                section_chunks = fixed_size_chunks(section_text)

            for chunk in section_chunks:
                chunks.append({
                    "section": section["section"],
                    "text": chunk
                })

    else:
        # Fallback for documents without headings
        for chunk in fixed_size_chunks(text):
            chunks.append({
                "section": "Full document",
                "text": chunk
            })

    return chunks


# ---------------- INGESTION PIPELINE ----------------

def ingest_documents():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        print(f"ERROR: Document folder not found: {DATA_DIR}")
        print("Create data/raw and add your source documents.")
        return

    files = sorted(
        file_path
        for file_path in DATA_DIR.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print("No supported HTML, Markdown, or TXT files found.")
        return

    records = []
    skipped_files = 0

    for file_path in files:
        plain_text, sections = extract_text(file_path)

        # Skip empty documents
        if not plain_text.strip() and not sections:
            print(f"Skipped empty file: {file_path}")
            skipped_files += 1
            continue

        document_chunks = split_into_chunks(
            plain_text,
            sections
        )

        relative_source = file_path.relative_to(DATA_DIR).as_posix()

        for chunk_index, chunk in enumerate(document_chunks):
            chunk_text = clean_text(chunk["text"])

            # Skip empty chunks
            if not chunk_text:
                continue

            if len(chunk_text) < MIN_CHUNK_SIZE:
                continue

            unique_string = (
                f"{relative_source}:{chunk_index}:{chunk_text}"
            )

            chunk_id = hashlib.sha256(
                unique_string.encode("utf-8")
            ).hexdigest()

            records.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": relative_source,
                    "section": chunk["section"],
                    "chunk_index": chunk_index
                }
            })

        print(
            f"Processed: {relative_source} "
            f"({len(document_chunks)} chunks)"
        )

    # Write JSON Lines output
    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # Report statistics
    chunk_lengths = [
        len(record["text"])
        for record in records
    ]

    print("\n========== INGESTION REPORT ==========")
    print(f"Documents discovered: {len(files)}")
    print(f"Empty/unreadable documents skipped: {skipped_files}")
    print(f"Chunks generated: {len(records)}")

    if chunk_lengths:
        print(
            "Average chunk size: "
            f"{statistics.mean(chunk_lengths):.2f} characters"
        )
        print(
            "Shortest chunk: "
            f"{min(chunk_lengths)} characters"
        )
        print(
            "Longest chunk: "
            f"{max(chunk_lengths)} characters"
        )

    print(f"Output file: {OUTPUT_FILE.resolve()}")
    print("======================================")

    if not records:
        print("WARNING: No usable chunks were generated.")


if __name__ == "__main__":
    ingest_documents()