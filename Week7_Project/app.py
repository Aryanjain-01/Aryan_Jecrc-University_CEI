from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag_engine import FinanceRAG, index_document, clear_index, get_chunk_count, delete_document_index, extract_kpis
from storage import add_document, clear_documents, delete_document, load_documents
import document_parser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
SAMPLE_DIR = ROOT / "sample_reports"

app = FastAPI(title="Finance RAG project Assignment week 7")


class DocumentUpload(BaseModel):
    name: str = "Untitled report"
    text: str = ""
    fileBase64: str | None = None


class QueryRequest(BaseModel):
    query: str
    topK: int = 5


def clean_report_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def decode_upload(payload: DocumentUpload) -> tuple[str, str]:
    name = payload.name
    text = payload.text
    encoded_file = payload.fileBase64

    if encoded_file:
        file_bytes = base64.b64decode(encoded_file)
        if name.lower().endswith(".pdf"):
            text = document_parser.extract_pdf_content(file_bytes)
            if not text.strip():
                raise ValueError("Could not extract any text or tables from the PDF.")
        else:
            text = file_bytes.decode("utf-8", errors="ignore")

    text = clean_report_text(text)
    if len(text) < 80:
        raise ValueError("Please provide a longer finance report text or upload a readable text/PDF file.")
    return name, text


@app.get("/")
async def get_root():
    """Serve the main HTML page."""
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/documents")
async def get_documents():
    """Get list of all documents."""
    documents = load_documents()
    return [
        {
            "id": document.id,
            "name": document.name,
            "characters": len(document.text),
            "kpis": getattr(document, "kpis", None),
        }
        for document in documents
    ]


@app.get("/api/stats")
async def get_stats():
    """Get statistics about documents and chunks."""
    documents = load_documents()
    return {"documents": len(documents), "chunks": get_chunk_count()}


@app.get("/api/insights")
async def get_insights():
    """Get company insights: positives and expenditure breakdown."""
    documents = load_documents()
    if not documents:
        return {
            "positives": [],
            "expenditure": {},
        }
    rag = FinanceRAG(documents)
    return rag.get_company_insights()


@app.post("/api/documents", status_code=201)
async def create_document(payload: DocumentUpload):
    """Create/upload a new document."""
    try:
        name, text = decode_upload(payload)
        
        # Automatically extract KPIs using the AI model
        kpis = extract_kpis(text)
        
        document = add_document(name, text, kpis=kpis)
        index_document(document)
        return {
            "id": document.id,
            "name": document.name,
            "characters": len(document.text),
            "kpis": document.kpis,
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/api/query")
async def query_documents(payload: QueryRequest):
    """Query the documents using RAG."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Question is required.")
    
    try:
        rag = FinanceRAG(load_documents())
        return rag.answer(query, top_k=payload.topK)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/api/load-sample")
async def load_sample():
    """Load sample reports."""
    clear_documents()
    clear_index()
    for file_path in sorted(SAMPLE_DIR.glob("*.txt")):
        doc = add_document(file_path.name, file_path.read_text(encoding="utf-8"))
        index_document(doc)
    documents = load_documents()
    return {"loaded": len(documents)}


@app.post("/api/clear")
async def clear_all():
    """Clear all documents."""
    clear_documents()
    clear_index()
    return {"ok": True}


@app.delete("/api/documents/{document_id}")
async def delete_doc(document_id: str):
    """Delete a specific document."""
    if delete_document(document_id):
        delete_document_index(document_id)
        return {"ok": True}
    else:
        raise HTTPException(status_code=404, detail="Document not found.")


# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
