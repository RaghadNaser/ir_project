# services/api_gateway/main.py

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Import settings
import sys
sys.path.append('..')
from config.settings import *

app = FastAPI(
    title="API Gateway with Web UI",
    description="Main API Gateway for Information Retrieval System",
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

# Setup paths for templates and static
app.mount("/static", StaticFiles(directory="services/api_gateway/static"), name="static")
templates = Jinja2Templates(directory="services/api_gateway/templates")

# Configuration loaded from config/settings.py

USER_QUERIES_FILE = "data/vectors/user_queries.tsv"

class SearchRequest(BaseModel):
    query: str
    dataset: str
    method: str = "hybrid"
    top_k: int = 10

class ServiceManager:
    """Service manager for handling service communications"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=90.0)  # Increased timeout for agent service
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
    """Home page with optional services"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "datasets": DATASETS,
        "representations": REPRESENTATIONS,
        "hybrid_methods": HYBRID_METHODS
    })

@app.get("/agent", response_class=HTMLResponse)
def agent_page(request: Request):
    """Agent conversational interface"""
    return templates.TemplateResponse("agent.html", {
        "request": request,
        "datasets": DATASETS
    })

@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    """Chat interface for agent service"""
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "datasets": DATASETS
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
    
    # Connect to database to get document information
    db_path = "data/ir_database_combined.db"
    conn = None
    cursor = None
    
    try:
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            print(f"✅ Connected to database: {db_path}")
            
            # Check available tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            print(f"📋 Available tables: {table_names}")
        else:
            print(f"❌ Database file not found: {db_path}")
    except Exception as e:
        print(f"❌ Database connection error: {e}")
    
    for result in results.get("results", []):
        doc_id = result.get("doc_id", result.get("id", ""))
        print(f"📄 Processing result with doc_id: {doc_id}")
        
        # Get document information from database
        if cursor and doc_id:
            print(f"🔍 Looking for document {doc_id} in {dataset} dataset")
            try:
                if dataset.lower() == "argsme":
                    cursor.execute("""
                        SELECT conclusion, premises_texts, source_title, topic
                        FROM argsme_raw 
                        WHERE doc_id = ?
                    """, (doc_id,))
                    row = cursor.fetchone()
                    if row:
                        conclusion = row[0] if row[0] else ""
                        premises = row[1] if row[1] else ""
                        source_title = row[2] if row[2] else ""  # Fixed index
                        topic = row[3] if row[3] else ""         # Fixed index
                        
                        print(f"✅ Found ARGSME document: conclusion={len(conclusion)} chars, premises={len(premises)} chars")
                        
                        # Create title from source_title or conclusion
                        if source_title:
                            title = source_title
                        elif conclusion:
                            # Take first 100 characters of conclusion
                            title = conclusion[:100] + "..." if len(conclusion) > 100 else conclusion
                        else:
                            title = f"Argument {doc_id}"
                        
                        # Create preview text
                        preview_text = conclusion[:200] + "..." if len(conclusion) > 200 else conclusion
                        if not preview_text and premises:
                            preview_text = premises[:200] + "..." if len(premises) > 200 else premises
                        
                        result["title"] = title
                        result["preview_text"] = preview_text
                        result["source_title"] = source_title
                        result["topic"] = topic
                    else:
                        print(f"❌ ARGSME document {doc_id} not found")
                        result["title"] = f"Document {doc_id}"
                        result["preview_text"] = "No preview available"
                
                elif dataset.lower() == "wikir":
                    cursor.execute("""
                        SELECT text 
                        FROM wikir_docs 
                        WHERE doc_id = ?
                    """, (doc_id,))
                    row = cursor.fetchone()
                    if row:
                        text = row[0] if row[0] else ""
                        
                        print(f"✅ Found WIKIR document: text={len(text)} chars")
                        
                        # Create title from first part of text
                        if text:
                            # Take first 100 characters as title
                            title = text[:100] + "..." if len(text) > 100 else text
                            # Take first 200 characters as preview
                            preview_text = text[:200] + "..." if len(text) > 200 else text
                        else:
                            title = f"Page {doc_id}"
                            preview_text = "No preview available"
                        
                        result["title"] = title
                        result["preview_text"] = preview_text
                    else:
                        print(f"❌ WIKIR document {doc_id} not found")
                        result["title"] = f"Page {doc_id}"
                        result["preview_text"] = "No preview available"
                
                else:
                    result["title"] = f"Document {doc_id}"
                    result["preview_text"] = "No preview available"
                    
            except Exception as e:
                print(f"Error getting document info for {doc_id}: {e}")
                result["title"] = f"Document {doc_id}"
                result["preview_text"] = "Error loading preview"
        else:
            result["title"] = f"Document {doc_id}"
            result["preview_text"] = "No preview available"
        
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
    
    # Close database connection
    if conn:
        conn.close()
    
    return enhanced_results

@app.post("/", response_class=HTMLResponse)
async def search_form(
    request: Request,
    dataset: str = Form(...),
    representation: str = Form(...),
    query: str = Form(...),
    top_k: int = Form(10),
    method: str = Form("sequential"),
    first_stage_k: int = Form(2000),
    tfidf_weight: float = Form(0.4),
    embedding_weight: float = Form(0.6),
    enable_topic_detection: str = Form(None),
    topic_max_topics: str = Form(None),
    topic_min_score: str = Form(None),
    enable_query_suggestion: str = Form(None),
    suggestion_method: str = Form(None),
    suggestion_count: str = Form(None),
    enable_vector_store: str = Form(None),
    vector_index_type: str = Form(None),
    vector_performance: str = Form(None)
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
    
    # Handle optional services
    topics = None
    suggestions = None
    vector_results = None
    
    if enable_topic_detection and ENABLE_TOPIC_DETECTION:
        try:
            topic_data = {
                "query": query,
                "dataset": dataset,
                "max_topics": int(topic_max_topics) if topic_max_topics else 5,
                "min_relevance_score": float(topic_min_score) if topic_min_score else 0.1
            }
            topic_result = await service_manager.call_service(
                "topic_detection", "detect-topics",
                method="POST",
                data=topic_data
            )
            topics = topic_result.get("detected_topics", [])
        except Exception as e:
            print(f"Topic detection error: {e}")
            topics = []
    
    if enable_query_suggestion:
        try:
            suggestion_data = {
                "query": query,
                "dataset": dataset,
                "method": suggestion_method or "hybrid",
                "top_k": int(suggestion_count) if suggestion_count else 8
            }
            suggestion_result = await service_manager.call_service(
                "query_suggestions", "suggest",
                method="POST",
                data=suggestion_data
            )
            suggestions = suggestion_result.get("suggestions", [])
        except Exception as e:
            print(f"Query suggestion error: {e}")
            suggestions = []
    
    if enable_vector_store:
        try:
            vector_data = {
                "query": query,
                "dataset": dataset,
                "top_k": int(top_k),
                "index_type": vector_index_type or "auto",
                "performance": vector_performance or "balanced"
            }
            vector_response = requests.post(f"http://localhost:8008/search", json=vector_data, timeout=10)
            if vector_response.status_code == 200:
                vector_result = vector_response.json()
                vector_results = vector_result.get("results", [])
        except Exception as e:
            print(f"Vector store error: {e}")
            vector_results = []

    # Debug: Print results structure
    print(f"Results type: {type(results)}")
    if results and len(results) > 0:
        print(f"First result type: {type(results[0])}")
        print(f"First result: {results[0]}")
    
    # Ensure results is a list of dictionaries
    if results and len(results) > 0 and not isinstance(results[0], dict):
        print("Converting results to proper format")
        # Convert list format to dict format if needed
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                formatted_results.append({
                    "doc_id": result[0],
                    "score": result[1],
                    "rank": i + 1
                })
            else:
                formatted_results.append({
                    "doc_id": str(result),
                    "score": 0.0,
                    "rank": i + 1
                })
        results = formatted_results

    # Enhance results with document information (title, preview, etc.)
    if results:
        if ENABLE_RESULT_ENHANCEMENT:
            try:
                enhanced_results = await enhance_search_results({"results": results}, query, dataset)
                results = enhanced_results
                print(f"Enhanced {len(results)} results with full document information")
            except Exception as e:
                print(f"Error enhancing results: {e}")
                # Fallback to simple titles
                for result in results:
                    doc_id = result.get("doc_id", result.get("id", ""))
                    if doc_id:
                        result["title"] = f"Document {doc_id}"
                        result["preview_text"] = f"Document ID: {doc_id}"
                    else:
                        result["title"] = "Unknown Document"
                        result["preview_text"] = "No preview available"
        else:
            # Fast mode - just use doc_id as title
            for result in results:
                doc_id = result.get("doc_id", result.get("id", ""))
                if doc_id:
                    # Create simple title from doc_id
                    if doc_id.startswith("doc_"):
                        result["title"] = f"Document {doc_id[4:]}"
                    else:
                        result["title"] = f"Document {doc_id}"
                    result["preview_text"] = f"Document ID: {doc_id}"
                else:
                    result["title"] = "Unknown Document"
                    result["preview_text"] = "No preview available"
            
            print(f"Processed {len(results)} results with simple titles (fast mode)")

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
        "method": method,
        "top_k": top_k,
        "first_stage_k": first_stage_k,
        "tfidf_weight": tfidf_weight,
        "embedding_weight": embedding_weight,
        "execution_time": execution_time,
        "performance_stats": performance_stats,
        "search_keywords": search_keywords,
        "total_results": len(results),
        "topics": topics,
        "suggestions": suggestions,
        "vector_results": vector_results
    })

@app.get("/suggestions")
async def get_query_suggestions(dataset: str = "argsme", num_suggestions: int = 5, query: str = ""):
    """Get query suggestions"""
    try:
        # If no query provided, get popular suggestions
        if not query:
            # Return some default suggestions based on dataset
            default_suggestions = {
                "argsme": [
                    "climate change", "vaccination", "gun control", "abortion", 
                    "immigration", "education", "healthcare", "economy"
                ],
                "wikir": [
                    "artificial intelligence", "machine learning", "data science", 
                    "programming", "technology", "computer science", "algorithms", "software"
                ]
            }
            
            suggestions = default_suggestions.get(dataset, ["data", "information", "search", "query"])
            return {"suggestions": [{"query": s} for s in suggestions[:num_suggestions]]}
        
        # If query provided, get suggestions from service
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
        # Return default suggestions on error
        default_suggestions = {
            "argsme": ["climate change", "vaccination", "gun control"],
            "wikir": ["artificial intelligence", "machine learning", "data science"]
        }
        suggestions = default_suggestions.get(dataset, ["data", "information"])
        return {"suggestions": [{"query": s} for s in suggestions[:num_suggestions]]}

@app.get("/suggestions/methods")
async def get_suggestion_methods():
    """Get available suggestion methods"""
    try:
        methods = await service_manager.call_service(
            "query_suggestions", "methods",
            method="GET"
        )
        return methods
    except Exception as e:
        print(f"Error getting suggestion methods: {e}")
        return {
            "methods": [
                {
                    "name": "hybrid",
                    "description": "Hybrid: Semantic + Popular + Autocomplete",
                    "best_for": "Most effective and smart suggestions"
                },
                {
                    "name": "semantic",
                    "description": "Semantic (Embedding-based) Suggestions",
                    "best_for": "Smart meaning-based suggestions"
                },
                {
                    "name": "popular",
                    "description": "Most Popular User Queries",
                    "best_for": "Discovering common user searches"
                },
                {
                    "name": "autocomplete",
                    "description": "Auto-complete from User Queries",
                    "best_for": "Completing user search queries"
                },
                {
                    "name": "semantic_terms",
                    "description": "Semantic Terms from Documents",
                    "best_for": "Finding relevant terms from document content"
                },
                {
                    "name": "hybrid_terms",
                    "description": "Hybrid Terms: Document Terms + User Queries",
                    "best_for": "Best of both worlds - document terms and user patterns"
                }
            ]
        }

@app.post("/suggestions/extract-terms")
async def extract_document_terms(request: Request):
    """Extract terms from documents"""
    try:
        data = await request.json()
        dataset = data.get("dataset", "argsme")
        top_k = data.get("top_k", 1000)
        
        result = await service_manager.call_service(
            "query_suggestions", "extract-terms",
            method="POST",
            data={"dataset": dataset, "top_k": top_k}
        )
        
        return result
    except Exception as e:
        print(f"Error extracting terms: {e}")
        return {
            "dataset": data.get("dataset", "argsme"),
            "terms_count": 0,
            "terms": [],
            "status": "error",
            "error": str(e)
        }

@app.post("/suggestions/build-embeddings")
async def build_term_embeddings(request: Request):
    """Build embedding vectors for terms"""
    try:
        data = await request.json()
        dataset = data.get("dataset", "argsme")
        top_k = data.get("top_k", 1000)
        
        result = await service_manager.call_service(
            "query_suggestions", "build-term-embeddings",
            method="POST",
            data={"dataset": dataset, "top_k": top_k}
        )
        
        return result
    except Exception as e:
        print(f"Error building embeddings: {e}")
        return {
            "dataset": data.get("dataset", "argsme"),
            "status": "error",
            "error": str(e)
        }

@app.post("/suggestions")
async def post_smart_suggestions(request: Request):
    """POST endpoint for smart suggestions with JSON data"""
    try:
        data = await request.json()
        query = data.get("query", "")
        dataset = data.get("dataset", "argsme")
        method = data.get("method", "hybrid")
        top_k = data.get("top_k", 8)
        include_metadata = data.get("include_metadata", True)
        
        # Call the query suggestions service
        suggestions = await service_manager.call_service(
            "query_suggestions", "suggest",
            method="POST",
            data={
                "query": query,
                "dataset": dataset,
                "method": method,
                "top_k": top_k,
                "include_metadata": include_metadata
            }
        )
        
        return suggestions
        
    except Exception as e:
        print(f"Error getting smart suggestions: {e}")
        # Return empty suggestions on error
        return {
            "suggestions": [],
            "method": data.get("method", "hybrid"),
            "dataset": data.get("dataset", "argsme"),
            "query": data.get("query", ""),
            "count": 0,
            "error": str(e)
        }

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

@app.post("/topics/detect")
async def detect_topics(request: Request):
    """Detect topics in a query"""
    try:
        data = await request.json()
        query = data.get("query", "")
        dataset = data.get("dataset", "argsme")
        max_topics = data.get("max_topics", 10)
        min_relevance_score = data.get("min_relevance_score", 0.1)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        result = await service_manager.call_service(
            "topic_detection", "detect-topics",
            method="POST",
            data={
                "query": query,
                "dataset": dataset,
                "max_topics": max_topics,
                "min_relevance_score": min_relevance_score
            }
        )
        
        return {
            "success": True,
            "data": result,
            "message": f"Detected {len(result.get('detected_topics', []))} topics"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to detect topics"
        }

@app.get("/topics/suggestions")
async def get_topic_suggestions(
    dataset: str = "argsme",
    limit: int = 20,
    category: Optional[str] = None
):
    """Get topic suggestions"""
    try:
        params = {"dataset": dataset, "limit": limit}
        if category:
            params["category"] = category
            
        result = await service_manager.call_service(
            "topic_detection", "suggest-topics",
            method="GET",
            data=params
        )
        
        return {
            "success": True,
            "data": result,
            "message": f"Retrieved {len(result.get('suggestions', []))} topic suggestions"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to get topic suggestions"
        }

@app.get("/topics/model-info")
async def get_topic_model_info(dataset: str = "argsme"):
    """Get topic model information"""
    try:
        result = await service_manager.call_service(
            "topic_detection", "model-info",
            method="GET",
            data={"dataset": dataset}
        )
        
        return {
            "success": True,
            "data": result,
            "message": f"Model info retrieved for {dataset} dataset"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to get model info"
        }

@app.get("/topics/datasets")
async def get_topic_datasets():
    """Get available datasets for topic detection"""
    try:
        result = await service_manager.call_service(
            "topic_detection", "datasets",
            method="GET"
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Datasets information retrieved"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to get datasets info"
        }

@app.get("/topics/health")
async def get_topic_service_health():
    """Get topic detection service health"""
    try:
        result = await service_manager.call_service(
            "topic_detection", "health",
            method="GET"
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Topic detection service health check completed"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to check topic detection service health"
        }

@app.get("/statistics")
async def get_system_statistics():
    """Get system statistics"""
    try:
        stats = {}
        
        # Get statistics from each service
        for service_name in ["tfidf", "embedding", "hybrid", "topic_detection", "agent"]:
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

@app.post("/agent/chat")
async def agent_chat(request: Request):
    """Proxy endpoint for agent chat"""
    try:
        body = await request.json()
        print(f"Agent chat request: {body}")
        
        # Create a client with longer timeout specifically for agent service
        async with httpx.AsyncClient(timeout=120.0) as client:  # 3 minutes timeout
            service_url = SERVICE_URLS.get("agent")
            if not service_url:
                raise HTTPException(status_code=404, detail="Agent service not found")
            
            url = f"{service_url}/chat"
            print(f"Calling agent service at: {url}")
            
            response = await client.post(url, json=body)
            print(f"Agent service response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Agent service response: {result}")
                return result
            else:
                print(f"Agent service error: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Agent service error: {response.text}"
                )
        
    except httpx.TimeoutException:
        print("Agent service timeout")
        raise HTTPException(status_code=504, detail="Agent service timeout - request took too long")
    except Exception as e:
        print(f"Agent chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/sessions")
async def get_agent_sessions():
    """Get active agent sessions"""
    try:
        return await service_manager.call_service(
            "agent", "sessions",
            method="GET"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/agent/sessions/{session_id}")
async def delete_agent_session(session_id: str):
    """Delete agent session"""
    try:
        return await service_manager.call_service(
            "agent", f"sessions/{session_id}",
            method="DELETE"
        )
    except Exception as e:
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

@app.get("/document/{doc_id}", response_class=HTMLResponse)
def view_document(request: Request, doc_id: str, dataset: str = "argsme"):
    """View document details"""
    try:
        # Connect to database
        db_path = "data/ir_database_combined.db"
        if not os.path.exists(db_path):
            return templates.TemplateResponse("document.html", {
                "request": request,
                "error": "Database not found",
                "dataset": dataset
            })
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query document based on dataset
        if dataset.lower() == "argsme":
            cursor.execute("""
                SELECT doc_id, conclusion, premises_texts, source_title, topic, acquisition
                FROM argsme_raw 
                WHERE doc_id = ?
            """, (doc_id,))
            row = cursor.fetchone()
            if row:
                document = {
                    "doc_id": doc_id,
                    "conclusion": row[1] if row[1] else "No conclusion available",
                    "premises_texts": row[2] if row[2] else "No premises available",
                    "source_title": row[3] if row[3] else "Unknown Source",
                    "topic": row[4] if row[4] else "General",
                    "acquisition": row[5] if row[5] else "Unknown Date",
                    "text": f"{row[1] if row[1] else ''} {row[2] if row[2] else ''}".strip(),
                    "title": f"Argument {doc_id}",
                }
            else:
                document = None
        
        elif dataset.lower() == "wikir":
            cursor.execute("""
                SELECT doc_id, text 
                FROM wikir_docs 
                WHERE doc_id = ?
            """, (doc_id,))
            row = cursor.fetchone()
            if row:
                document = {
                    "doc_id": doc_id,
                    "text": row[1] if row[1] else "No text available",
                    "title": f"Page {doc_id}"
                }
            else:
                document = None
        
        else:
            document = None
        
        conn.close()
        
        if document is None:
            error = f"Document '{doc_id}' not found in {dataset} dataset"
        else:
            error = None
        
        return templates.TemplateResponse("document.html", {
            "request": request,
            "document": document,
            "dataset": dataset,
            "error": error
        })
        
    except Exception as e:
        return templates.TemplateResponse("document.html", {
            "request": request,
            "error": f"Error loading document: {str(e)}",
            "dataset": dataset
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)