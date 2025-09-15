# NCD CDSS FastAPI Backend

A prototype Clinical Decision Support System for non-communicable diseases using FastAPI.

## Features

- Upload NCD guideline PDFs.
- Extract and chunk guideline text.
- Analyze patient cases using guideline excerpts and optional LLM (OpenAI API integration).
- Simple web interface.

## Setup

1. Clone the repo.
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Copy `.env.example` to `.env` and set your keys/paths.
4. Start the server:
    ```
    uvicorn main:app --reload
    ```
5. Visit [http://localhost:8000](http://localhost:8000) in your browser.

## .env File

```
OPENAI_API_KEY=your_openai_key_here
UPLOAD_DIR=./uploaded_pdfs
KNOWLEDGE_DIR=./knowledge
GITHUB_REPO_URL=https://github.com/YOUR_USERNAME/YOUR_REPO
```

## Project structure

- `main.py` — FastAPI app.
- `templates/` — Jinja2 HTML templates.
- `static/` — Stylesheets and static files.
- `uploaded_pdfs/` — Uploaded guideline PDFs.
- `knowledge/` — Extracted guideline chunks.

---
This tool supports but does not replace clinical judgment.
