from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Any
import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified Search Service",
    description="Unified service with optional vector store usage",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
EMBEDDING_SERVICE_URL = "http://localhost:8004"  # Original embedding service
VECTOR_STORE_SERVICE_URL = "http://localhost:8007"  # Vector store service
SEARCH_SERVICE_URL = "http://localhost:8004"  # Original search service

class UnifiedSearchRequest(BaseModel):
    dataset: str
    query: str
    top_k: int = 10
    use_vector_store: bool = True  # الخيار للتحكم في استخدام vector store
    index_type: Optional[str] = "auto"

class UnifiedSearchResponse(BaseModel):
    results: List[Tuple[str, float]]
    query: str
    dataset: str
    method_used: str
    use_vector_store: bool
    total_results: int
    total_time: float
    performance_stats: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Unified Search Service",
        "version": "1.0.0",
        "description": "Search with optional vector store usage",
        "options": {
            "use_vector_store: true": "Uses embedding → vector store pipeline (fastest)",
            "use_vector_store: false": "Uses traditional search (slower but different approach)"
        },
        "endpoints": {
            "/search": "POST - Unified search with vector store option",
            "/health": "GET - Health check"
        }
    }

@app.post("/search", response_model=UnifiedSearchResponse)
async def unified_search(request: UnifiedSearchRequest):
    """Unified search with optional vector store usage"""
    try:
        start_time = time.time()
        
        # Validate input
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if request.dataset not in ["argsme", "wikir"]:
            raise HTTPException(status_code=400, detail="Dataset must be 'argsme' or 'wikir'")
        
        if request.use_vector_store:
            # استخدام Vector Store Pipeline
            method_used = "embedding_to_vector_store"
            
            logger.info(f"🔄 Using Vector Store Pipeline for: '{request.query}'")
            
            # Step 1: Get embedding
            embed_start = time.time()
            embed_response = requests.post(
                f"{EMBEDDING_SERVICE_URL}/embed",
                json={"text": request.query},
                timeout=30
            )
            
            if embed_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to get embedding")
            
            embed_data = embed_response.json()
            embed_time = time.time() - embed_start
            
            # Step 2: Search with vector store (with score normalization)
            search_start = time.time()
            search_response = requests.post(
                f"{VECTOR_STORE_SERVICE_URL}/search",
                json={
                    "dataset": request.dataset,
                    "query_vector": embed_data["embedding"],
                    "top_k": request.top_k,
                    "index_type": request.index_type,
                    "score_normalization": "minmax"  # Use minmax for stable [0,1] range
                },
                timeout=30
            )
            
            if search_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Vector store search failed")
            
            search_data = search_response.json()
            search_time = time.time() - search_start
            
            results = search_data["results"]
            
            performance_stats = {
                "embedding_time": embed_time,
                "vector_search_time": search_time,
                "embedding_dimension": embed_data["dimension"],
                "index_type": search_data["index_type"],
                "vector_store_stats": search_data.get("performance_stats", {}),
                "pipeline": "embedding → vector_store"
            }
            
        else:
            # استخدام Traditional Search
            method_used = "traditional_search"
            
            logger.info(f"🔄 Using Traditional Search for: '{request.query}'")
            
            # Use the original search service
            search_response = requests.post(
                f"{SEARCH_SERVICE_URL}/search",
                json={
                    "dataset": request.dataset,
                    "query": request.query,
                    "top_k": request.top_k
                },
                timeout=60
            )
            
            if search_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Traditional search failed")
            
            search_data = search_response.json()
            results = search_data["results"]
            
            performance_stats = {
                "query_time": search_data.get("query_time", 0),
                "total_docs": search_data.get("total_docs", 0),
                "method": search_data.get("method", "Standard"),
                "gpu_used": search_data.get("gpu_used", False),
                "pipeline": "traditional_search"
            }
        
        total_time = time.time() - start_time
        
        # Log performance comparison
        logger.info(f"✅ Search completed in {total_time:.3f}s using {method_used}")
        
        return UnifiedSearchResponse(
            results=results,
            query=request.query,
            dataset=request.dataset,
            method_used=method_used,
            use_vector_store=request.use_vector_store,
            total_results=len(results),
            total_time=total_time,
            performance_stats=performance_stats
        )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Service communication error: {e}")
        raise HTTPException(status_code=500, detail=f"Service communication failed: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unified search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check all dependent services
    services_status = {}
    
    try:
        # Check embedding service
        embed_response = requests.get(f"{EMBEDDING_SERVICE_URL}/health", timeout=5)
        services_status["embedding_service"] = embed_response.status_code == 200
    except:
        services_status["embedding_service"] = False
    
    try:
        # Check vector store service
        vector_response = requests.get(f"{VECTOR_STORE_SERVICE_URL}/health", timeout=5)
        services_status["vector_store_service"] = vector_response.status_code == 200
    except:
        services_status["vector_store_service"] = False
    
    try:
        # Check traditional search service
        search_response = requests.get(f"{SEARCH_SERVICE_URL}/health", timeout=5)
        services_status["search_service"] = search_response.status_code == 200
    except:
        services_status["search_service"] = False
    
    all_healthy = all(services_status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "service": "Unified Search Service",
        "dependent_services": services_status,
        "vector_store_available": services_status.get("embedding_service", False) and services_status.get("vector_store_service", False),
        "traditional_search_available": services_status.get("search_service", False),
        "timestamp": time.time()
    }

@app.get("/compare/{dataset}")
async def compare_methods(dataset: str, query: str = "machine learning", top_k: int = 5):
    """Compare vector store vs traditional search performance"""
    try:
        results = {}
        
        # Test vector store method
        try:
            vector_request = UnifiedSearchRequest(
                dataset=dataset,
                query=query,
                top_k=top_k,
                use_vector_store=True
            )
            vector_result = await unified_search(vector_request)
            results["vector_store"] = {
                "method": vector_result.method_used,
                "time": vector_result.total_time,
                "results_count": vector_result.total_results,
                "performance": vector_result.performance_stats
            }
        except Exception as e:
            results["vector_store"] = {"error": str(e)}
        
        # Test traditional method
        try:
            traditional_request = UnifiedSearchRequest(
                dataset=dataset,
                query=query,
                top_k=top_k,
                use_vector_store=False
            )
            traditional_result = await unified_search(traditional_request)
            results["traditional"] = {
                "method": traditional_result.method_used,
                "time": traditional_result.total_time,
                "results_count": traditional_result.total_results,
                "performance": traditional_result.performance_stats
            }
        except Exception as e:
            results["traditional"] = {"error": str(e)}
        
        return {
            "query": query,
            "dataset": dataset,
            "comparison": results
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008) 