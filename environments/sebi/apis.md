# Xyne Document Ingestion API

**Endpoint:** `POST http://localhost:3000/files/upload`
**Purpose:** Upload PDFs/DOCXs/PPTXs/images to Vespa for RAG retrieval
**Date:** 2026-05-17

---

## Quick Start

```bash
JWT="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJybC11c2VyQHRlc3QubG9jYWwiLCJ3b3Jrc3BhY2VJZCI6Indrc3BfdGVzdDEyMyIsInJvbGUiOiJVc2VyIiwiaWF0IjoxNzc4ODQ3NzMxLCJleHAiOjE3ODE0Mzk3MzF9.70aZY17xH36IPwWVvKvcnZ-wTUMf6C-0W6S-7kMURxI"

curl -X POST http://localhost:3000/files/upload \
  -H "Authorization: Bearer $JWT" \
  -F "file=@report.pdf" \
  -F "datasourceName=my-kb" \
  -F "flag=creation"
```

**Fields:**
- `file` - The file (PDF, DOCX, PPTX, XLSX, CSV, TXT, JPG, PNG)
- `datasourceName` - Knowledge base name (creates if new)
- `flag` - `"creation"` (first file) or `"addition"` (subsequent files)

**Response:**
```json
{
  "success": true,
  "message": "Successfully processed 1 file(s)",
  "dataSourceResults": [{"docId": "doc_xxx", "fileName": "report.pdf"}]
}
```

---

## What Happens

```
POST /files/upload
    -> api/files.ts (receives file)
    -> api/dataSource.ts (creates KB if needed)
    -> integrations/dataSource/index.ts (detects file type)
    -> FileProcessorService.processFile() (chunks text + describes images)
    -> queue/fileProcessor.ts (stores in Vespa with embeddings)
```

**Chunking:** PDFs go through Gemini Vision (<=40 pages) -> Docling -> OCR -> PDFJS cascade. Text files chunked by paragraphs. Images described via LLM.

**Embedding:** `bge-small-en-v1.5` (384-dim) inside Vespa container.

**Searchable after:** ~5-30 seconds.

---

## Bulk Upload Script

```python
#!/usr/bin/env python3
import os, requests
from pathlib import Path

BASE_URL = "http://localhost:3000"
JWT = "your-jwt-here"
KB_NAME = "my-corpus"
HEADERS = {"Authorization": f"Bearer {JWT}"}
SUPPORTED = {'.pdf', '.docx', '.pptx', '.xlsx', '.csv', '.txt', '.md', '.jpg', '.png'}

def upload(filepath: str, kb: str, flag: str):
    with open(filepath, 'rb') as f:
        resp = requests.post(
            f"{BASE_URL}/files/upload",
            headers=HEADERS,
            files={'file': (os.path.basename(filepath), f)},
            data={'datasourceName': kb, 'flag': flag},
            timeout=300,
        )
    return resp.json()

dir_path = Path("/path/to/documents")
files = [f for f in dir_path.iterdir() if f.suffix.lower() in SUPPORTED]
flag = "creation"

for f in files:
    try:
        result = upload(str(f), KB_NAME, flag)
        print(f"{'OK' if result.get('success') else 'FAIL'}: {f.name}")
        flag = "addition"
    except Exception as e:
        print(f"ERR: {f.name} - {e}")
        flag = "addition"
```

**Usage:** `python3 bulk_upload.py`

---

## Limits

| Type | Max Size |
|------|----------|
| General file | 40MB |
| Image | 5MB |
| PDF | 100MB |
| DOCX/PPTX/XLSX | 50MB |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| 401 Unauthorized | JWT expired, get new token |
| File rejected | Check extension is supported, check size |
| Processing slow | Normal for large PDFs with images (Gemini API call) |
| Not searchable | Wait 30s, check `curl localhost:8081/search/?query=test` |
| Image captions missing | Check `LITELLM_BASE_URL` env var is set |

---

## Verify Ingestion

```bash
# Count documents
curl "localhost:8081/search/?query=sddocname:file&hits=0" | jq '.root.fields.totalCount'

# Search
curl "localhost:8081/search/?query=remote+work&hits=3" | jq '.root.children[].fields.title'
```

---

## Key Files (for debugging)

| File | What it does |
|------|-------------|
| `xyne/server/api/files.ts` | HTTP handler for `/files/upload` |
| `xyne/server/integrations/dataSource/index.ts` | Routes file to correct processor |
| `xyne/server/services/fileProcessor.ts` | Orchestrates chunking |
| `xyne/server/lib/pdfProcessor.ts` | PDF processing with fallback chain |
| `xyne/server/queue/fileProcessor.ts` | Stores chunks in Vespa |
| `xyne/server/vespa/schemas/file.sd` | Vespa document schema |
