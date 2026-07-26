import fitz  # type: ignore

def extract_pdf_content(file_bytes: bytes) -> str:
    """Extract text from a PDF using PyMuPDF for maximum memory efficiency."""
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text("text") for page in document)
    except Exception as e:
        print(f"PyMuPDF failed: {e}")
        return ""
