# doc-llm-processor

A command-line tool that converts collections of documents (PDFs, Word, PowerPoint, text files) into chunked, structured JSON suitable for feeding into LLMs.

You point it at a folder of documents. It extracts the text, splits it into overlapping chunks that respect sentence boundaries, and writes the result as JSON/JSONL with full source metadata. The output is designed to be loaded into LLM context windows, vector stores, or RAG pipelines.

## Why this exists

If you have a pile of PDFs and Office documents that you want an LLM to reason over, you need to get the text out, chunk it sensibly, and keep track of where each piece came from. This tool does that in one step.

The fast processor uses PyMuPDF for native text extraction from born-digital PDFs (no OCR overhead), with optional OCR fallback for scanned documents. It processes files in parallel and is typically 100-1000x faster than OCR-first approaches.

## Requirements

- Python 3.8+
- For PDF processing: no extra system dependencies needed (PyMuPDF handles it)
- For scanned/image-based PDFs (optional OCR fallback): [Tesseract](https://github.com/tesseract-ocr/tesseract) and [Poppler](https://poppler.freedesktop.org/)

## Installation

```
git clone https://github.com/visser23/doc-llm-processor.git
cd doc-llm-processor
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Quick start

Put your documents in a folder (e.g. `src/my-documents/`), then run:

```
python fast_processor.py --input-dir "src/my-documents" --output-dir "output"
```

This creates an `output/` folder containing:

| File | Contents |
|------|----------|
| `knowledge_base.json` | Full knowledge base: metadata + all chunks + file index |
| `chunks.json` | Just the chunks array (convenient for direct LLM upload) |
| `knowledge_base_chunks.jsonl` | One chunk per line (for streaming/incremental loading) |
| `processing_metadata.json` | Processing stats and per-file index |
| `processing_summary.csv` | Spreadsheet-friendly overview of all processed files |

## Where to put your source files

Create a folder anywhere and pass it with `--input-dir`. The tool recursively scans for supported file types:

- `.pdf` — native text extraction, OCR fallback for scans
- `.docx` — paragraph extraction via python-docx
- `.pptx` — slide-by-slide text extraction
- `.txt`, `.md` — read as-is

The `src/` directory in this repo is gitignored, so you can use it locally without risk of committing your documents.

## Command-line options

```
python fast_processor.py [OPTIONS]

Required:
  --input-dir, -i     Folder containing your documents

Optional:
  --output-dir, -o    Where to write output (default: processed_knowledge_base)
  --chunk-size, -c    Max characters per chunk (default: 1000)
  --overlap           Character overlap between chunks (default: 200)
  --workers, -w       Parallel worker count (default: CPU count - 1)
  --no-parallel       Process files one at a time
  --no-ocr-fallback   Skip OCR for scanned documents
  --format, -f        Output format: json, jsonl, or both (default: both)
```

## Output format

Each chunk looks like this:

```json
{
  "chunk_id": "quarterly_report_p002_c001",
  "source_file": "reports/quarterly_report.pdf",
  "file_hash": "a1b2c3d4e5f6...",
  "chunk_index": 5,
  "text": "Revenue increased 12% year-over-year driven by...",
  "metadata": {
    "filename": "quarterly_report.pdf",
    "file_directory": "reports",
    "file_size_bytes": 245760,
    "page_number": 2,
    "processed_timestamp": "2026-01-15T10:30:00",
    "extraction_method": "native"
  }
}
```

Key fields:
- `chunk_id` — unique, derived from filename + page + chunk number
- `source_file` — relative path from your input directory
- `file_hash` — MD5 of the source file (for change detection)
- `text` — the actual content
- `metadata.extraction_method` — `native` or `ocr_fallback`

## Querying the output

There's a built-in query tool for searching and exploring processed output:

```
# Search across all chunks
python query_knowledge_base.py search "revenue forecast"

# List all processed files
python query_knowledge_base.py list

# Show stats
python query_knowledge_base.py stats

# Get all chunks from a specific file
python query_knowledge_base.py content "quarterly_report.pdf"

# Export filtered results as formatted text
python query_knowledge_base.py export --query "revenue" --output context.txt
```

## Using the output with LLMs

### Direct upload

Upload `chunks.json` to ChatGPT, Claude, or similar tools as a file attachment. The JSON structure is self-describing.

### In code

```python
import json

with open("output/knowledge_base.json") as f:
    kb = json.load(f)

# Get all chunks
chunks = kb["chunks"]

# Filter by source file
report_chunks = [c for c in chunks if "report" in c["source_file"].lower()]

# Build context string
context = "\n\n".join(c["text"] for c in report_chunks[:20])
```

### With a vector store (e.g. LangChain + FAISS)

```python
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

with open("output/knowledge_base.json") as f:
    kb = json.load(f)

docs = [
    Document(
        page_content=chunk["text"],
        metadata={"source": chunk["source_file"], "page": chunk["metadata"].get("page_number")}
    )
    for chunk in kb["chunks"]
]

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
results = vectorstore.similarity_search("quarterly revenue", k=5)
```

## Legacy processor

`process_documents.py` is an older processor that uses the [Unstructured.io](https://unstructured.io/) library. It supports more file types (HTML, Excel) and does OCR-first processing, but is significantly slower and has heavier dependencies. To use it, uncomment the Unstructured lines in `requirements.txt` and install.

```
python process_documents.py --input-dir "src/my-documents" --output-dir "output"
```

## Chunk sizing guide

The default chunk size of 1000 characters (~250 tokens) works well for most use cases. Adjust based on your needs:

- **Smaller chunks (500 chars)**: better for precise retrieval in vector search
- **Larger chunks (2000+ chars)**: better for maintaining document context, fewer chunks to manage
- **Overlap (default 200 chars)**: prevents information loss at chunk boundaries

## Licence

MIT
