# doc-llm-processor

A command-line tool that converts collections of documents (PDFs, Word, PowerPoint, text files) into chunked, structured JSON suitable for feeding into LLMs.

You point it at a folder of documents. It extracts the text, splits it into overlapping chunks that respect sentence boundaries, and writes the output as a set of organized files with temporal ordering and source grouping. The output is designed to be uploaded directly to LLM chat interfaces, or loaded into vector stores and RAG pipelines.

## Why this exists

If you have a pile of PDFs and Office documents that you want an LLM to reason over, you need to get the text out, chunk it sensibly, and keep track of where each piece came from. Dumping everything into one massive file creates problems: LLMs lose track of what came from where, recent information gets buried among old content, and unrelated topics bleed into each other.

This tool solves that by:
- Extracting text fast (PyMuPDF native extraction, no OCR overhead for born-digital PDFs)
- Detecting document dates (content scanning for large files, file metadata for smaller ones)
- Grouping output by source folder so related documents stay together
- Ordering everything newest-first so the LLM prioritises current information
- Capping output at a small number of files (default 10: 1 index + up to 9 content files + 1 JSONL)

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

Put your documents in a folder (e.g. `src/my-project/`), then run:

```
python fast_processor.py --input-dir "src/my-project" --output-dir "output"
```

## Where to put your source files

Create a folder anywhere and pass it with `--input-dir`. The tool recursively scans for supported file types:

- `.pdf` -- native text extraction, OCR fallback for scans
- `.docx` -- paragraph extraction via python-docx
- `.pptx` -- slide-by-slide text extraction
- `.txt`, `.md` -- read as-is

The `src/` directory in this repo is gitignored, so you can use it locally without risk of committing your documents.

## Output structure

By default the tool writes an **organized** output to a clean folder:

```
output/
  00_index.json             <-- read this first: what's here, dates, navigation
  01_most_recent_group.json <-- content file (newest source folder)
  02_next_group.json        <-- content file
  ...
  all_chunks.jsonl          <-- flat file for programmatic/RAG use
```

### Index file (`00_index.json`)

The index is designed for an LLM to read first. It describes the full output: what each content file contains, the date range of its documents, and how to navigate.

```json
{
  "description": "Knowledge base from 47 documents in 'project-docs'",
  "processed": "2026-06-08T12:00:00",
  "total_documents": 47,
  "total_chunks": 892,
  "date_range": { "earliest": "2024-06-15", "latest": "2026-03-15" },
  "navigation": "Content files are numbered and ordered newest-first...",
  "files": [
    {
      "filename": "01_reports.json",
      "group": "reports",
      "document_count": 8,
      "chunk_count": 156,
      "date_range": { "earliest": "2025-01-10", "latest": "2026-03-15" },
      "documents": ["latest_update.pdf", "q4_review.pdf"]
    }
  ]
}
```

### Content files (`01_*.json`, `02_*.json`, ...)

Each content file groups documents from the same source subfolder. Within each file, documents are ordered newest-first, and chunks within each document are in page order.

```json
{
  "group": "reports",
  "document_count": 8,
  "chunk_count": 156,
  "date_range": { "earliest": "2025-01-10", "latest": "2026-03-15" },
  "documents": [
    {
      "source_file": "reports/latest_update.pdf",
      "filename": "latest_update.pdf",
      "document_date": "2026-03-15",
      "page_count": 12,
      "chunk_count": 24,
      "chunks": [ ... ]
    }
  ]
}
```

### Flat JSONL (`all_chunks.jsonl`)

One chunk per line, all chunks from the run. Use this for vector store ingestion, search indexing, or any programmatic pipeline where you want raw access to every chunk.

### Chunk structure

Each chunk carries its source metadata:

```json
{
  "chunk_id": "quarterly_report_p002_c001",
  "source_file": "reports/quarterly_report.pdf",
  "file_hash": "a1b2c3d4e5f6...",
  "chunk_index": 5,
  "text": "Revenue increased 12% year-over-year...",
  "metadata": {
    "filename": "quarterly_report.pdf",
    "file_directory": "reports",
    "file_size_bytes": 245760,
    "page_number": 2,
    "document_date": "2026-03-15",
    "processed_timestamp": "2026-06-08T12:00:00",
    "extraction_method": "native"
  }
}
```

- `document_date` -- extracted from content for large files (>512KB), from file modification date for smaller files. Used for temporal ordering.
- `extraction_method` -- `native` (PyMuPDF direct text) or `ocr_fallback` (Unstructured.io OCR).

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
  --max-groups        Max content files in organized output (default: 9)
  --flat              Use legacy flat output (single knowledge_base.json) instead of organized
  --format, -f        Output format for --flat mode: json, jsonl, or both (default: both)
```

## How document dates are detected

The tool tries to figure out when each document is from, so it can order output newest-first:

1. **Large files (>512KB)**: scans the first 10,000 characters of extracted text for date patterns -- ISO dates, UK/US long dates, month-year references. Takes the most recent date found. This handles Confluence exports and other large generated documents where dates are embedded in the content.
2. **Smaller files**: uses the file's last-modified timestamp from the filesystem.

This is a heuristic. It won't be perfect for every document, but it gets the ordering right in most cases and is far better than no temporal signal at all.

## Using the output with LLMs

### Direct upload to ChatGPT / Claude

Upload the index file (`00_index.json`) and the content files you need. The LLM can read the index first to understand what's available, then reference specific content files. You don't need to upload all files if you only care about a subset.

### In code

```python
import json

with open("output/00_index.json") as f:
    index = json.load(f)

# Find the most recent content file
newest_file = index["files"][0]["filename"]

with open(f"output/{newest_file}") as f:
    group = json.load(f)

for doc in group["documents"]:
    print(f"{doc['filename']} ({doc['document_date']}): {doc['chunk_count']} chunks")
```

### With a vector store

Load `all_chunks.jsonl` for programmatic ingestion:

```python
import json
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

docs = []
with open("output/all_chunks.jsonl") as f:
    for line in f:
        chunk = json.loads(line)
        docs.append(Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source_file"],
                "page": chunk["metadata"].get("page_number"),
                "date": chunk["metadata"].get("document_date"),
            }
        ))

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
```

## Querying the output

There's a built-in query tool for searching and exploring processed output:

```
python query_knowledge_base.py search "revenue forecast"
python query_knowledge_base.py list
python query_knowledge_base.py stats
python query_knowledge_base.py content "quarterly_report.pdf"
python query_knowledge_base.py export --query "revenue" --output context.txt
```

Note: the query tool works with flat output (`--flat` mode). For organized output, use the JSON files directly or load `all_chunks.jsonl`.

## Legacy processor

`process_documents.py` is an older processor that uses [Unstructured.io](https://unstructured.io/). It supports more file types (HTML, Excel) and does OCR-first processing, but is significantly slower and has heavier dependencies. To use it, uncomment the Unstructured lines in `requirements.txt` and install.

## Chunk sizing guide

The default chunk size of 1000 characters (~250 tokens) works well for most use cases:

- **Smaller chunks (500 chars)**: better for precise retrieval in vector search
- **Larger chunks (2000+ chars)**: better for maintaining document context, fewer chunks to manage
- **Overlap (default 200 chars)**: prevents information loss at chunk boundaries

## Licence

MIT
