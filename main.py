import os
import io
import json
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from difflib import SequenceMatcher
import openai

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Config
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploaded_pdfs")
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "./knowledge")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

GITHUB_REPO_URL = os.getenv("GITHUB_REPO_URL", "https://github.com/YOUR_USERNAME/YOUR_REPO")

app = FastAPI(title="NCD CDSS Prototype")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for p in range(len(reader.pages)):
        try:
            page = reader.pages[p]
            text = page.extract_text()
            if text:
                texts.append(text)
        except Exception:
            continue
    return "\n\n".join(texts)

def simple_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        # Try to break at a sentence boundary ahead
        if end < length:
            for i in range(end, min(length, end + 200)):
                if text[i] in ".\n":
                    chunk = text[start:i+1]
                    end = i+1
                    break
        chunks.append(chunk.strip())
        start = max(end - overlap, end)
    return [c for c in chunks if c]

def save_chunks_for_file(filename: str, chunks: List[str]):
    base = os.path.splitext(os.path.basename(filename))[0]
    out_path = os.path.join(KNOWLEDGE_DIR, f"{base}_chunks.json")
    payload = {"source_filename": os.path.basename(filename), "chunks": chunks}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path

def load_all_knowledge():
    """Return a dict mapping source -> list of chunks"""
    res = {}
    for fname in os.listdir(KNOWLEDGE_DIR):
        if fname.endswith("_chunks.json"):
            with open(os.path.join(KNOWLEDGE_DIR, fname), "r", encoding="utf-8") as f:
                j = json.load(f)
                res[j.get("source_filename", fname)] = j.get("chunks", [])
    return res

def rank_chunks_by_similarity(chunks: List[str], query: str, top_k: int = 5):
    scored = []
    for i, chunk in enumerate(chunks):
        score = SequenceMatcher(None, query.lower(), chunk.lower()).ratio()
        scored.append((i, score, chunk))
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# Models
class CaseIn(BaseModel):
    patient_age: Optional[str] = ""
    patient_gender: Optional[str] = ""
    chief_complaint: str
    history: Optional[str] = ""
    vitals: Optional[str] = ""
    investigations: Optional[str] = ""
    comorbidities: Optional[str] = ""

# Routes
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    uploaded = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    knowledge = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith("_chunks.json")]
    return templates.TemplateResponse("index.html", {"request": request, "uploaded": uploaded, "knowledge": knowledge})

@app.post("/upload-guideline")
async def upload_guideline(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"ok": False, "message": "Only PDF allowed."}, status_code=400)
    filename = os.path.basename(file.filename)
    contents = await file.read()
    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(contents)
    text = extract_text_from_pdf_bytes(contents)
    if not text.strip():
        return JSONResponse({"ok": False, "message": "No text could be extracted from that PDF."}, status_code=400)
    chunks = simple_chunk_text(text)
    save_chunks_for_file(save_path, chunks)
    return {"ok": True, "filename": filename, "stored_chunks": len(chunks)}

@app.post("/upload-guidelines-batch")
async def upload_guidelines_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            results.append({"filename": file.filename, "ok": False, "message": "Not a PDF"})
            continue
        filename = os.path.basename(file.filename)
        contents = await file.read()
        save_path = os.path.join(UPLOAD_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(contents)
        text = extract_text_from_pdf_bytes(contents)
        if not text.strip():
            results.append({"filename": filename, "ok": False, "message": "No text extracted"})
            continue
        chunks = simple_chunk_text(text)
        save_chunks_for_file(save_path, chunks)
        results.append({"filename": filename, "ok": True, "stored_chunks": len(chunks)})
    return {"results": results}

@app.get("/knowledge-list")
def knowledge_list():
    return load_all_knowledge()

@app.post("/analyze-case")
async def analyze_case(case: CaseIn):
    kb = load_all_knowledge()
    if not kb:
        return JSONResponse({"ok": False, "message": "No guideline knowledge found. Upload PDFs first."}, status_code=400)
    query_text = (
        f"Age: {case.patient_age}\n"
        f"Gender: {case.patient_gender}\n"
        f"Chief complaint: {case.chief_complaint}\n"
        f"History: {case.history}\n"
        f"Vitals: {case.vitals}\n"
        f"Investigations: {case.investigations}\n"
        f"Comorbidities: {case.comorbidities}\n"
    )
    # Retrieve top chunks across all documents
    retrieved = []
    for source, chunks in kb.items():
        top = rank_chunks_by_similarity(chunks, query_text, top_k=5)
        for idx, score, chunk in top:
            retrieved.append({"source": source, "chunk_index": idx, "score": score, "text": chunk})
    retrieved = sorted(retrieved, key=lambda x: x["score"], reverse=True)[:8]
    if OPENAI_API_KEY:
        system_prompt = (
            "You are an AI Clinical Decision Support System specialized in non-communicable diseases (NCDs). "
            "Use ONLY the guideline passages provided below as the authoritative source. "
            "Produce structured output with these headings: Diagnosis/Differential, Recommended Investigations, "
            "Management Plan (pharmacologic/non-pharmacologic), Monitoring & Follow-up, Red Flags/Referral Criteria. "
            "For each recommendation include the source filename and chunk index. Finish with: "
            "This tool supports but does not replace clinical judgment."
        )
        guideline_sections = []
        for r in retrieved:
            guideline_sections.append(f"--- SOURCE: {r['source']} CHUNK: {r['chunk_index']} ---\n{r['text']}\n")
        prompt_user = (
            "GUIDELINE PASSAGES (use only these):\n\n"
            + "\n\n".join(guideline_sections)
            + "\n\nPATIENT CASE:\n"
            + query_text
            + "\n\nINSTRUCTIONS: Provide the structured output as JSON with keys: diagnosis, investigations, management, monitoring, red_flags, references"
        )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_user}
                ],
                max_tokens=900,
                temperature=0.0
            )
            llm_text = resp["choices"][0]["message"]["content"]
        except Exception as e:
            llm_text = f"[LLM call failed: {str(e)}]\n\nRetrieved guideline excerpts returned instead."
        return {"ok": True, "case_summary": query_text, "retrieved": retrieved, "llm_output": llm_text}
    return {"ok": True, "case_summary": query_text, "retrieved": retrieved, "llm_output": None, "note": "Set OPENAI_API_KEY to enable AI summarization."}

@app.get("/status")
def status():
    uploaded = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    knowledge = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith("_chunks.json")]
    return {"ok": True, "uploaded_pdfs": uploaded, "knowledge_files": knowledge}

@app.get("/github-link")
def github_link():
    return {"github_repo": GITHUB_REPO_URL}
