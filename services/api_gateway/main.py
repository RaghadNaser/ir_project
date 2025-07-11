# services/api_gateway/main.py

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import sqlite3
import os
from datetime import datetime

app = FastAPI(title="API Gateway with Web UI")

# Setup paths for templates and static
app.mount("/static", StaticFiles(directory="services/api_gateway/static"), name="static")
templates = Jinja2Templates(directory="services/api_gateway/templates")

# Service map
SERVICE_URLS = {
    "tfidf": "http://localhost:8003/search",
    "embedding": "http://localhost:8004/search",
    "hybrid": "http://localhost:8005/search"
}

# Dataset list (update according to your files)
DATASETS = ["argsme", "wikir"]
REPRESENTATIONS = [
    ("tfidf", "TF-IDF"),
    ("embedding", "Embedding"),
    ("hybrid", "Hybrid (Enhanced)")
]

# New hybrid search methods
HYBRID_METHODS = [
    ("parallel", "Parallel Fusion"),
    ("serial_tfidf_first", "Serial TF-IDF First"),
    ("serial_embedding_first", "Serial Embedding First")
]

USER_QUERIES_FILE = "data/vectors/user_queries.tsv"
def save_user_query(query: str, dataset: str):
    if not query.strip():
        return
    print(f"Saving user query: '{query}' for dataset: '{dataset}'")
    os.makedirs(os.path.dirname(USER_QUERIES_FILE), exist_ok=True)
    with open(USER_QUERIES_FILE, "a", encoding="utf-8") as f:
        f.write(f"{dataset}\t{query.strip()}\t{datetime.utcnow().isoformat()}\n")
    print(f"Query saved successfully to {USER_QUERIES_FILE}")

class SearchRequest(BaseModel):
    dataset: str
    query: str
    top_k: int = 10
    representation: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "datasets": DATASETS,
        "representations": REPRESENTATIONS,
        "hybrid_methods": HYBRID_METHODS,
        "results": None,
        "query": "",
        "selected_dataset": DATASETS[0],
        "selected_representation": REPRESENTATIONS[0][0],
        "selected_method": "parallel",
        "top_k": 10,
        "tfidf_weight": 0.4,
        "embedding_weight": 0.6
    })

@app.post("/", response_class=HTMLResponse)
def search(
    request: Request,
    dataset: str = Form(...),
    representation: str = Form(...),
    query: str = Form(...),
    top_k: int = Form(10),
    method: str = Form("parallel"),
    first_stage_k: int = Form(2000),
    tfidf_weight: float = Form(0.4),
    embedding_weight: float = Form(0.6),
    enable_topic_detection: str = Form(None)
):
    save_user_query(query, dataset)
    url = SERVICE_URLS.get(representation)
    payload = {
        "dataset": dataset,
        "query": query,
        "top_k": int(top_k)
    }
    if representation == "hybrid":
        payload["method"] = method
        payload["first_stage_k"] = int(first_stage_k)
        payload["tfidf_weight"] = float(tfidf_weight)
        payload["embedding_weight"] = float(embedding_weight)
    else:
        payload["mode"] = "fusion" if representation == "hybrid" else None
    results = []
    error = None
    execution_time = 0.0
    performance_stats = {}
    search_keywords = []
    if url:
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                response_data = response.json()
                results = response_data.get("results", [])
                execution_time = response_data.get("execution_time", 0.0)
                performance_stats = response_data.get("performance_stats", {})
                if "error" in performance_stats and not error:
                    error = f"Hybrid Service Error: {performance_stats['error']}"
                # استخراج نصوص أعلى 10 نتائج
                top_docs = []
                for r in results[:10]:
                    if isinstance(r, dict):
                        doc_text = r.get("text") or r.get("processed_text")
                        if doc_text:
                            top_docs.append(str(doc_text))
                    elif isinstance(r, (list, tuple)) and len(r) > 2:
                        top_docs.append(str(r[2]))
                print("Top docs for keyword extraction:", top_docs)
                if top_docs:
                    try:
                        print("Calling /extract_keywords with", len(top_docs), "docs")
                        keywords_resp = requests.post(
                            "http://localhost:8006/extract_keywords",
                            json={"documents": top_docs, "top_k": 10},
                            timeout=10
                        )
                        print("Keyword API status:", keywords_resp.status_code)
                        print("Keyword API response:", keywords_resp.text)
                        if keywords_resp.status_code == 200:
                            keywords_data = keywords_resp.json()
                            search_keywords = keywords_data.get("keywords", [])
                    except Exception as e:
                        print("Keyword extraction error:", e)
            else:
                error = f"Service error: {response.status_code}"
        except Exception as e:
            error = str(e)
    else:
        error = "Invalid representation"
    return templates.TemplateResponse("index.html", {
        "request": request,
        "datasets": DATASETS,
        "representations": REPRESENTATIONS,
        "hybrid_methods": HYBRID_METHODS,
        "results": results,
        "query": query,
        "selected_dataset": dataset,
        "selected_representation": representation,
        "selected_method": method,
        "top_k": top_k,
        "error": error,
        "execution_time": execution_time,
        "performance_stats": performance_stats,
        "first_stage_k": first_stage_k,
        "tfidf_weight": tfidf_weight,
        "embedding_weight": embedding_weight,
        "dataset": dataset,
        "search_keywords": search_keywords,
        "enable_topic_detection": False
    })

@app.get("/document/{dataset}/{doc_id}", response_class=HTMLResponse)
async def get_document(request: Request, dataset: str, doc_id: str):
    db_path = "data/ir_database_combined.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if dataset == "argsme":
        cur.execute("SELECT * FROM argsme_raw WHERE doc_id=?", (doc_id,))
        row = cur.fetchone()
        columns = [desc[0] for desc in cur.description]
    elif dataset == "wikir":
        cur.execute("SELECT * FROM wikir_docs WHERE doc_id=?", (doc_id,))
        row = cur.fetchone()
        columns = [desc[0] for desc in cur.description]
    else:
        conn.close()
        return HTMLResponse("Unknown dataset", status_code=400)
    conn.close()
    if row:
        doc = dict(zip(columns, row))
        return templates.TemplateResponse("document.html", {"request": request, "doc": doc, "dataset": dataset})
    else:
        return HTMLResponse("Document not found", status_code=404)

@app.post("/suggest_api")
async def suggest_api(request: Request):
    data = await request.json()
    # Use the new smart service
    try:
        resp = requests.post("http://localhost:8010/suggest", json=data, timeout=10)
        return JSONResponse(content=resp.json())
    except Exception as e:
        return JSONResponse(content={"suggestions": [], "error": str(e)}, status_code=500)

@app.post("/topic_detect_api")
async def topic_detect_api(request: Request):
    data = await request.json()
    try:
        resp = requests.post("http://localhost:8006/detect_topic", json=data, timeout=10)
        return JSONResponse(content=resp.json())
    except Exception as e:
        return JSONResponse(content={"error": f"Topic detection service unavailable: {str(e)}"}, status_code=500)