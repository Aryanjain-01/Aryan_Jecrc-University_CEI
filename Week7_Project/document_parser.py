import fitz  # type: ignore
import pymupdf4llm

def extract_pdf_content(file_bytes: bytes) -> str:
    """Extract text and markdown tables from a PDF using pymupdf4llm."""
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
        md_text = pymupdf4llm.to_markdown(doc=document)
        return md_text
    except Exception as e:
        print(f"pymupdf4llm failed: {e}")
        return ""
