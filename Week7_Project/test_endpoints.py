import asyncio
from fastapi.testclient import TestClient
from app import app
from storage import clear_documents

client = TestClient(app)

def test_all():
    clear_documents()
    
    # 1. Test empty stats
    res = client.get("/api/stats")
    print("GET /api/stats (empty):", res.status_code, res.text)
    
    # 2. Test insights
    res = client.get("/api/insights")
    print("GET /api/insights (empty):", res.status_code, res.text)
    
    # 3. Load sample
    res = client.post("/api/load-sample")
    print("POST /api/load-sample:", res.status_code, res.text)
    
    # 4. Stats with documents
    res = client.get("/api/stats")
    print("GET /api/stats (loaded):", res.status_code, res.text)
    
    # 5. Query
    res = client.post("/api/query", json={"query": "revenue", "topK": 5})
    print("POST /api/query:", res.status_code, res.text[:200])

test_all()
