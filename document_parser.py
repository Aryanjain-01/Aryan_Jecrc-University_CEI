import io
import fitz  # type: ignore
import pdfplumber

def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert a 2D list table from pdfplumber to Markdown format."""
    if not table:
        return ""
    
    clean_table = []
    for row in table:
        clean_row = [str(cell).replace("\n", " ").strip() if cell is not None else "" for cell in row]
        if any(clean_row):  # Only keep rows that have at least some text
            clean_table.append(clean_row)
            
    if not clean_table:
        return ""
        
    num_cols = max(len(row) for row in clean_table)
    
    for row in clean_table:
        while len(row) < num_cols:
            row.append("")
            
    md_lines = []
    
    # Header row
    md_lines.append("| " + " | ".join(clean_table[0]) + " |")
    # Separator row
    md_lines.append("|" + "|".join(["---"] * num_cols) + "|")
    
    # Data rows
    for row in clean_table[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
        
    return "\n".join(md_lines) + "\n\n"

def extract_pdf_content(file_bytes: bytes) -> str:
    """Extract text and tables from a PDF using pdfplumber, falling back to PyMuPDF."""
    try:
        full_text = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # We want to extract text, but pdfplumber's extract_text can scramble tables.
                # However, for RAG, having both the raw text and the explicitly formatted Markdown table is perfectly fine.
                page_text = page.extract_text() or ""
                
                tables = page.extract_tables()
                if tables:
                    table_md = f"\n\n### Formatted Tables (Page {page_num + 1}):\n\n"
                    for table in tables:
                        table_md += _table_to_markdown(table)
                    page_text += table_md
                    
                full_text.append(page_text)
                
        return "\n\n---\n\n".join(full_text)
        
    except Exception as e:
        print(f"pdfplumber failed: {e}")
        # Fast fallback to PyMuPDF
        try:
            document = fitz.open(stream=file_bytes, filetype="pdf")
            return "\n".join(page.get_text("text") for page in document)
        except Exception:
            return ""
