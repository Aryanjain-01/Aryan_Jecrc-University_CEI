from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
import json
from dataclasses import dataclass, asdict
from typing import Iterable

import chromadb
from sentence_transformers import SentenceTransformer, util


@dataclass
class Document:
    id: str
    name: str
    text: str
    kpis: dict[str, str] | None = None


@dataclass
class Chunk:
    id: str
    document_id: str
    document_name: str
    text: str
    chunk_index: int


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


def chunk_text(text: str, max_words: int = 180, overlap: int = 35) -> list[str]:
    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return []

    words = clean_text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)

    return chunks


def extract_financial_metrics(text: str) -> list[str]:
    patterns = [
        r"(revenue|net sales|sales|income|profit|ebitda|gross margin|operating margin|cash flow|free cash flow|assets|liabilities|debt|eps|earnings per share)[^.:\n]{0,90}?(\$?\d+(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|m|bn|%)?)",
        r"(\$?\d+(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|m|bn|%)?)\s+(revenue|net sales|sales|income|profit|ebitda|gross margin|operating margin|cash flow|free cash flow|assets|liabilities|debt)",
    ]

    findings: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrase = " ".join(part.strip() for part in match.groups())
            phrase = re.sub(r"\s+", " ", phrase)
            key = phrase.lower()
            if key not in seen:
                findings.append(phrase)
                seen.add(key)
    return findings[:10]


def extract_company_positives(text: str) -> list[str]:
    """Extract positive aspects and strengths of the company."""
    positive_keywords = [
        r"strong\s+(?:growth|revenue|cash flow|balance sheet|market position)",
        r"increased\s+(?:market share|profit|revenue|sales|efficiency)",
        r"successful\s+(?:launch|expansion|acquisition|product)",
        r"(?:record|high|strong)\s+(?:earnings|revenue|profit|margins)",
        r"expanded\s+(?:operations|customer base|geographic presence)",
        r"leading\s+(?:market|industry|position)",
        r"improved\s+(?:efficiency|margins|performance|profitability)",
        r"competitive\s+(?:advantage|position|edge)",
        r"strong\s+(?:brand|reputation|customer loyalty)",
        r"operational\s+excellence",
    ]

    positives: list[str] = []
    seen: set[str] = set()
    text_lower = text.lower()
    
    for pattern in positive_keywords:
        for match in re.finditer(pattern, text_lower):
            # Extract surrounding context
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            phrase = text[start:end].strip()
            phrase = re.sub(r"\s+", " ", phrase)
            
            key = phrase.lower()[:100]
            if key not in seen:
                positives.append(phrase[:150])
                seen.add(key)
    
    return positives[:8]


def extract_expenditure_breakdown(text: str) -> dict[str, list[str]]:
    """Extract company expenditure details by category."""
    expenditure_patterns = {
        "Employee Costs": [
            r"salaries?\s+(?:and)?\s+(?:wages?|benefits?|compensation)[^.]*?(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn)?",
            r"payroll\s+(?:expenses?|costs?)[^.]*?(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn)?",
            r"personnel\s+(?:expenses?|costs?)[^.]*?(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn)?",
        ],
        "Infrastructure & Technology": [
            r"(?:capital|infrastructure|technology)\s+(?:expenditure|investment|spending)[^.]*?(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn)?",
            r"(?:IT|information technology|computer|system)\s+(?:expenses?|costs?|spending)[^.]*?(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn)?",
            r"property.*equipment\s+(?:expenses?|costs?)[^.]*?(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn)?",
        ],
        "Research & Development": [
            r"(?:research|development|r&d|r & d)\s+(?:expenses?|spending|investment)[^.]*?(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn)?",
            r"innovation\s+(?:expenses?|costs?)[^.]*?(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn)?",
        ],
        "Marketing & Sales": [
            r"(?:marketing|advertising|sales)\s+(?:expenses?|spending|costs?)[^.]*?(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn)?",
            r"promotional\s+(?:expenses?|costs?)[^.]*?(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn)?",
        ],
        "Operational Costs": [
            r"operating\s+(?:expenses?|costs?)[^.]*?(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn)?",
            r"cost of\s+(?:revenue|goods sold|services)[^.]*?(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn)?",
            r"general.*administrative\s+(?:expenses?|costs?)[^.]*?(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn)?",
        ],
    }
    
    breakdown: dict[str, list[str]] = {}
    text_lower = text.lower()
    
    for category, patterns in expenditure_patterns.items():
        findings: list[str] = []
        seen: set[str] = set()
        
        for pattern in patterns:
            for match in re.finditer(pattern, text_lower):
                phrase = match.group(0).strip()
                phrase = re.sub(r"\s+", " ", phrase)
                key = phrase.lower()[:100]
                
                if key not in seen and len(phrase) > 10:
                    findings.append(phrase[:200])
                    seen.add(key)
        
        if findings:
            breakdown[category] = findings[:3]
    
    return breakdown


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


# Initialize semantic model globally to prevent reloading on every request
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="data/chroma")
collection = chroma_client.get_or_create_collection(name="finance_chunks")

def index_document(document: Document):
    chunks = chunk_text(document.text)
    if not chunks:
        return
    
    ids = []
    documents = []
    metadatas = []
    
    for i, text in enumerate(chunks):
        ids.append(f"{document.id}::chunk-{i+1}")
        documents.append(text)
        metadatas.append({
            "document_id": document.id,
            "document_name": document.name,
            "chunk_index": i + 1
        })
        
    # We use sentence-transformers to encode so we have consistent vectors
    embeddings = MODEL.encode(documents, convert_to_tensor=False).tolist()
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

def clear_index():
    global collection
    try:
        chroma_client.delete_collection("finance_chunks")
    except ValueError:
        pass
    collection = chroma_client.get_or_create_collection(name="finance_chunks")

def get_chunk_count() -> int:
    return collection.count()

def delete_document_index(document_id: str):
    collection.delete(where={"document_id": document_id})

class FinanceRAG:
    def __init__(self, documents: Iterable[Document]):
        self.documents = list(documents)
        self.model = MODEL

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if collection.count() == 0:
            return []

        query_embedding = self.model.encode(query, convert_to_tensor=False).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        ranked: list[RetrievalResult] = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                chunk = Chunk(
                    id=results['ids'][0][i],
                    document_id=meta['document_id'],
                    document_name=meta['document_name'],
                    text=results['documents'][0][i],
                    chunk_index=meta['chunk_index']
                )
                score = 1.0 / (1.0 + results['distances'][0][i])
                ranked.append(RetrievalResult(chunk=chunk, score=score))

        return ranked

    def answer(self, query: str, top_k: int = 5) -> dict:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return {
                "answer": "I could not find relevant evidence in the uploaded finance reports.",
                "metrics": [],
                "sources": [],
            }

        openai_answer = self._try_openai_answer(query, results)
        if openai_answer:
            answer_text = openai_answer
        else:
            answer_text = self._extractive_answer(query, results)

        combined_text = " ".join(result.chunk.text for result in results)
        return {
            "answer": answer_text,
            "metrics": extract_financial_metrics(combined_text),
            "sources": [
                {
                    **asdict(result.chunk),
                    "score": round(result.score, 4),
                    "preview": result.chunk.text[:420] + ("..." if len(result.chunk.text) > 420 else ""),
                }
                for result in results
            ],
        }

    def get_company_insights(self) -> dict:
        """Extract company positives and expenditure breakdown."""
        combined_text = " ".join(doc.text for doc in self.documents)
        
        return {
            "positives": extract_company_positives(combined_text),
            "expenditure": extract_expenditure_breakdown(combined_text),
        }

    def _extractive_answer(self, query: str, results: list[RetrievalResult]) -> str:
        query_vector = self.model.encode(query, convert_to_tensor=True)
        candidates: list[tuple[float, str, Chunk]] = []

        for result in results:
            sentences = split_sentences(result.chunk.text)
            if not sentences:
                continue
                
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
            similarities = util.cos_sim(query_vector, sentence_embeddings)[0]
            
            for sentence, score in zip(sentences, similarities):
                score_val = float(score)
                if score_val > 0.1:
                    candidates.append((score_val, sentence, result.chunk))

        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            first_sentences = split_sentences(results[0].chunk.text)
            first_sentence = first_sentences[0] if first_sentences else results[0].chunk.text
            candidates = [(0.0, first_sentence, results[0].chunk)]

        lines = []
        used_sentences: set[str] = set()
        for _, sentence, chunk in candidates:
            key = sentence.lower()
            if key in used_sentences:
                continue
            used_sentences.add(key)
            lines.append(f"{sentence} [{chunk.document_name}, chunk {chunk.chunk_index}]")
            if len(lines) == 4:
                break

        return " ".join(lines)

    def _try_openai_answer(self, query: str, results: list[RetrievalResult]) -> str | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        context = "\n\n".join(
            f"Source: {result.chunk.document_name}, chunk {result.chunk.chunk_index}\n{result.chunk.text}"
            for result in results
        )
        prompt = (
            "You are a finance report analysis assistant. Answer only from the supplied context. "
            "IMPORTANT: Use clear bullet points (pointers) for readability and present financial data/numbers prominently. "
            "Include citations in square brackets using the provided source and chunk labels. "
            "If the context is insufficient, say so.\n\n"
            f"Question: {query}\n\nContext:\n{context}"
        )
        body = json.dumps(
            {
                "model": os.getenv("OPENAI_MODEL", "qwen/qwen3-32b"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
        ).encode("utf-8")
        
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
        request = urllib.request.Request(
            base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "FiscalMind/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError):
            return None


def extract_kpis(text: str) -> dict[str, str] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    # We take the first 40,000 characters (~10,000 tokens) to stay within context limits
    # and because KPIs are usually in the first few pages of the report.
    context = text[:40000]

    prompt = (
        "You are an expert financial analyst. Extract the following KPIs from the provided document text:\n"
        "- Revenue\n"
        "- Profit\n"
        "- Assets\n"
        "- Liabilities\n"
        "- Promoter Holding\n"
        "- FII Holding\n"
        "- DII Holding\n"
        "- Public Holding\n"
        "- Operating Cash Flow\n"
        "- Investing Cash Flow\n"
        "- Financing Cash Flow\n\n"
        "If a metric is not found in the text, use 'N/A'. Keep values extremely concise (e.g., '$10.5B', '45%', '2,340 cr').\n"
        "Output ONLY a valid JSON object where the keys are the exact metric names above.\n\n"
        f"Document Text:\n{context}"
    )

    body = json.dumps(
        {
            "model": os.getenv("OPENAI_MODEL", "qwen/qwen3-32b"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
    ).encode("utf-8")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
    request = urllib.request.Request(
        base_url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "FiscalMind/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"].strip()
            # Clean markdown code blocks if the AI returns them
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
    except Exception as e:
        print(f"Error extracting KPIs: {e}")
        return None
