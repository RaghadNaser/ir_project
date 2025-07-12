# services/api_gateway/main.py

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import sqlite3
import os
from datetime import datetime
import asyncio
import httpx
from typing import Optional, Dict, Any

app = FastAPI(
    title="API Gateway with Web UI",
    description="Main API Gateway for Information Retrieval System",
    version="1.0.0"
)

# Setup paths for templates and static
app.mount("/static", StaticFiles(directory="services/api_gateway/static"), name="static")
templates = Jinja2Templates(directory="services/api_gateway/templates")

# Service URLs
SERVICE_URLS = {
    "preprocessing": "http://localhost:8002",
    "tfidf": "http://localhost:8003",
    "embedding": "http://localhost:8004",
    "hybrid": "http://localhost:8005",
    "topic_detection": "http://localhost:8006",
    "query_suggestions": "http://localhost:8010"
}

# Search service map
SEARCH_SERVICE_URLS = {
    "tfidf": "http://localhost:8003/search",
    "embedding": "http://localhost:8004/search",
    "hybrid": "http://localhost:8005/search"
}

# Dataset list
DATASETS = ["argsme", "wikir"]
REPRESENTATIONS = [
    ("tfidf", "TF-IDF"),
    ("embedding", "Embedding"),
    ("hybrid", "Hybrid (Enhanced)")
]

# Hybrid search methods
HYBRID_METHODS = [
    ("sequential", "Sequential Search"),
    ("parallel", "Parallel Search"),
    ("fusion", "Fusion Search")
]

USER_QUERIES_FILE = "data/vectors/user_queries.tsv"

class SearchRequest(BaseModel):
    query: str
    dataset: str
    method: str = "hybrid"
    top_k: int = 10

class ServiceManager:
    """Service manager for handling service communications"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.service_status = {}
    
    async def check_service_health(self, service_name: str) -> bool:
        """Check service health"""
        try:
            service_url = SERVICE_URLS.get(service_name)
            if not service_url:
                return False
            
            response = await self.client.get(f"{service_url}/health")
            return response.status_code == 200
        except Exception as e:
            print(f"Error checking service {service_name}: {e}")
            return False
    
    async def call_service(self, service_name: str, endpoint: str, 
                          method: str = "GET", data: Optional[Dict] = None) -> Dict:
        """Call a specific service"""
        try:
            service_url = SERVICE_URLS.get(service_name)
            if not service_url:
                raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
            
            url = f"{service_url}/{endpoint}"
            
            if method.upper() == "POST":
                response = await self.client.post(url, json=data or {})
            else:
                response = await self.client.get(url, params=data or {})
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Service {service_name} error: {response.text}"
                )
            
            return response.json()
        
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail=f"Service {service_name} timeout")
        except Exception as e:
            print(f"Error calling service {service_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

service_manager = ServiceManager()

def save_user_query(query: str, dataset: str):
    """Save user query to file"""
    if not query.strip():
        return
    print(f"Saving user query: '{query}' for dataset: '{dataset}'")
    os.makedirs(os.path.dirname(USER_QUERIES_FILE), exist_ok=True)
    with open(USER_QUERIES_FILE, "a", encoding="utf-8") as f:
        f.write(f"{dataset}\t{query.strip()}\t{datetime.utcnow().isoformat()}\n")
    print(f"Query saved successfully to {USER_QUERIES_FILE}")

@app.get("/health")
async def health_check():
    """Health check endpoint for API Gateway"""
    health_status = {}
    
    # Check all services
    for service_name in SERVICE_URLS.keys():
        health_status[service_name] = await service_manager.check_service_health(service_name)
    
    overall_health = all(health_status.values())
    
    return {
        "status": "healthy" if overall_health else "degraded",
        "service": "api_gateway",
        "version": "1.0.0",
        "services": health_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "datasets": DATASETS,
        "representations": REPRESENTATIONS,
        "hybrid_methods": HYBRID_METHODS
    })

@app.post("/search")
async def unified_search(search_request: SearchRequest):
    """Unified search endpoint"""
    try:
        query = search_request.query
        dataset = search_request.dataset
        method = search_request.method
        top_k = search_request.top_k
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Save user query
        save_user_query(query, dataset)
        
        # Preprocess query
        processed_query = await service_manager.call_service(
            "preprocessing", "preprocess",
            method="POST",
            data={"text": query}
        )
        
        # Execute search based on method
        if method == "tfidf":
            results = await service_manager.call_service(
                "tfidf", "search",
                method="POST",
                data={
                    "query": processed_query["processed_text"],
                    "dataset": dataset,
                    "top_k": top_k
                }
            )
        elif method == "embedding":
            results = await service_manager.call_service(
                "embedding", "search",
                method="POST",
                data={
                    "query": query,  # Use original query for embedding
                    "dataset": dataset,
                    "top_k": top_k
                }
            )
        elif method == "hybrid":
            results = await service_manager.call_service(
                "hybrid", "search",
                method="POST",
                data={
                    "query": query,
                    "dataset": dataset,
                    "top_k": top_k,
                    "method": "fusion"  # Default to fusion
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported search method")
        
        # Enhance results with topic detection
        enhanced_results = await enhance_search_results(results, query, dataset)
        
        return {
            "query": query,
            "processed_query": processed_query["processed_text"],
            "method": method,
            "dataset": dataset,
            "results": enhanced_results,
            "total_results": len(enhanced_results),
            "search_time": results.get("execution_time", 0)
        }
    
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def enhance_search_results(results: Dict, query: str, dataset: str) -> list:
    """Enhance search results with additional information"""
    enhanced_results = []
    
    for result in results.get("results", []):
        # Add topic detection if available
        try:
            topic_info = await service_manager.call_service(
                "topic_detection", "predict",
                method="POST",
                data={
                    "dataset": dataset,
                    "text": result.get("content", ""),
                    "top_k": 3
                }
            )
            result["topics"] = topic_info.get("topics", [])
        except Exception as e:
            print(f"Topic detection error: {e}")
            result["topics"] = []
        
        enhanced_results.append(result)
    
    return enhanced_results

@app.post("/", response_class=HTMLResponse)
def search_form(
    request: Request,
    dataset: str = Form(...),
    representation: str = Form(...),
    query: str = Form(...),
    top_k: int = Form(10),
    method: str = Form("sequential"),
    first_stage_k: int = Form(2000),
    tfidf_weight: float = Form(0.4),
    embedding_weight: float = Form(0.6),
    enable_topic_detection: str = Form(None)
):
    """Handle form-based search"""
    save_user_query(query, dataset)
    url = SEARCH_SERVICE_URLS.get(representation)
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
    
    results = []
    error = None
    execution_time = 0.0
    performance_stats = {}
    search_keywords = []
    
    try:
        if not url:
            raise HTTPException(status_code=400, detail="Invalid representation")
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            execution_time = data.get("execution_time", 0.0)
            performance_stats = data.get("performance_stats", {})
            
            # Extract keywords from query
            search_keywords = query.split()
            
        else:
            error = f"Search failed: {response.text}"
    except Exception as e:
        error = f"Search error: {str(e)}"
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "datasets": DATASETS,
        "representations": REPRESENTATIONS,
        "hybrid_methods": HYBRID_METHODS,
        "results": results,
        "error": error,
        "query": query,
        "dataset": dataset,
        "representation": representation,
        "execution_time": execution_time,
        "performance_stats": performance_stats,
        "search_keywords": search_keywords,
        "total_results": len(results)
    })

@app.get("/suggestions")
async def get_query_suggestions(query: str, dataset: str = "argsme", num_suggestions: int = 5):
    """Get query suggestions"""
    try:
        suggestions = await service_manager.call_service(
            "query_suggestions", "suggest",
            method="POST",
            data={
                "query": query,
                "dataset": dataset,
                "method": "hybrid",
                "top_k": num_suggestions
            }
        )
        
        return suggestions
    
    except Exception as e:
        print(f"Error getting suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topics/{dataset}")
async def get_dataset_topics(dataset: str):
    """Get dataset topics"""
    try:
        topics = await service_manager.call_service(
            "topic_detection", "topics",
            method="GET",
            data={"dataset": dataset}
        )
        
        return topics
    
    except Exception as e:
        print(f"Error getting topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_system_statistics():
    """Get system statistics"""
    try:
        stats = {}
        
        # Get statistics from each service
        for service_name in ["tfidf", "embedding", "hybrid", "topic_detection"]:
            try:
                service_stats = await service_manager.call_service(
                    service_name, "stats"
                )
                stats[service_name] = service_stats
            except Exception as e:
                print(f"Error getting stats from {service_name}: {e}")
                stats[service_name] = {"error": str(e)}
        
        return stats
    
    except Exception as e:
        print(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest_api")
async def suggest_api(request: Request):
    """API endpoint for suggestions"""
    data = await request.json()
    try:
        resp = await service_manager.call_service(
            "query_suggestions", "suggest",
            method="POST",
            data=data
        )
        return JSONResponse(content=resp)
    except Exception as e:
        return JSONResponse(content={"suggestions": [], "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)