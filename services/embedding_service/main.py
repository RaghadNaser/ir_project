# services/embedding_service/main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import sys
import os
import logging
import time
import gc
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.logging_config import setup_logging
from models.embedding_model import EmbeddingModel
from utils.gpu_utils import get_gpu_accelerator, cleanup_gpu_memory
import psutil
from sentence_transformers import SentenceTransformer
import numpy as np

# Setup logging
logger = setup_logging("embedding_service")

# Initialize FastAPI app
app = FastAPI(
    title="Optimized Embedding Service",
    description="GPU-accelerated embedding search service with FAISS indexing",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models dictionary
models = {}
gpu_accelerator = get_gpu_accelerator()

class SearchRequest(BaseModel):
    query: str
    dataset: str
    top_k: int = 10
    candidate_doc_ids: Optional[List[str]] = None
    use_faiss: bool = True
    index_type: str = "flat"  # "flat", "ivf", "hnsw"

class BatchSearchRequest(BaseModel):
    queries: List[str]
    dataset: str
    top_k: int = 10
    use_faiss: bool = True

class SearchResponse(BaseModel):
    results: List[tuple]
    query_time: float
    total_docs: int
    method: str
    gpu_used: bool

class BatchSearchResponse(BaseModel):
    results: List[List[tuple]]
    query_time: float
    total_queries: int
    method: str
    gpu_used: bool

class ModelInfoResponse(BaseModel):
    loaded_models: Dict[str, Any]
    gpu_info: Dict[str, Any]
    memory_stats: Dict[str, Any]

class EmbedRequest(BaseModel):
    text: str
    dataset: str = "argsme"

class EmbedResponse(BaseModel):
    embedding: List[float]
    text: str
    dataset: str
    dimension: int
    processing_time: float

def get_or_load_model(dataset: str) -> EmbeddingModel:
    """Get or load embedding model for dataset"""
    if dataset not in models:
        try:
            logger.info(f"Loading embedding model for dataset: {dataset}")
            model = EmbeddingModel(dataset)
            model.load_model()
            models[dataset] = model
            logger.info(f"Model loaded successfully for {dataset}")
        except Exception as e:
            logger.error(f"Failed to load model for {dataset}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model for {dataset}")
    
    return models[dataset]

@app.post("/search", response_model=SearchResponse)
async def search_embedding(request: SearchRequest):
    """Search using embedding model with GPU acceleration"""
    try:
        start_time = time.time()
        
        # Get model
        model = get_or_load_model(request.dataset)
        
        # Perform search
        if request.use_faiss and hasattr(model, 'search_faiss'):
            # Use FAISS if available
            results = model.search_faiss(
                request.query, 
                top_k=request.top_k,
                candidate_doc_ids=request.candidate_doc_ids
            )
            method = f"FAISS-{request.index_type}"
        else:
            # Standard search
            results = model.search(
                request.query, 
                top_k=request.top_k, 
                candidate_doc_ids=request.candidate_doc_ids
            )
            method = "Standard"
        
        query_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            query_time=query_time,
            total_docs=len(model.embeddings) if hasattr(model, 'embeddings') and model.embeddings is not None else 0,
            method=method,
            gpu_used=getattr(model, 'use_gpu', False)
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_search", response_model=BatchSearchResponse)
async def batch_search_embedding(request: BatchSearchRequest):
    """Batch search using embedding model with GPU optimization"""
    try:
        start_time = time.time()
        
        # Get model
        model = get_or_load_model(request.dataset)
        
        # Perform batch search
        if hasattr(model, 'batch_search'):
            # Use optimized batch search
            results = model.batch_search(request.queries, top_k=request.top_k)
            method = "Batch-Optimized"
        else:
            # Fallback to sequential search
            results = []
            for query in request.queries:
                query_results = model.search(query, top_k=request.top_k)
                results.append(query_results)
            method = "Sequential"
        
        query_time = time.time() - start_time
        
        return BatchSearchResponse(
            results=results,
            query_time=query_time,
            total_queries=len(request.queries),
            method=method,
            gpu_used=getattr(model, 'use_gpu', False)
        )
        
    except Exception as e:
        logger.error(f"Batch search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about loaded models and GPU status"""
    try:
        # Get model info
        model_info = {}
        for dataset, model in models.items():
            model_info[dataset] = {
                'loaded': model.loaded,
                'use_gpu': getattr(model, 'use_gpu', False),
                'embeddings_shape': getattr(model.embeddings, 'shape', None) if hasattr(model, 'embeddings') else None,
                'device': str(getattr(model, 'device', 'unknown'))
            }
        
        # Get GPU info
        gpu_info = gpu_accelerator.get_memory_stats()
        
        # Get memory stats for all models
        memory_stats = {}
        for dataset, model in models.items():
            if hasattr(model, 'get_memory_usage'):
                memory_stats[dataset] = model.get_memory_usage()
        
        # Add system memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_stats['system'] = {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024
        }
        
        return ModelInfoResponse(
            loaded_models=model_info,
            gpu_info=gpu_info,
            memory_stats=memory_stats
        )
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cleanup_gpu")
async def cleanup_gpu():
    """Clean up GPU memory"""
    try:
        # Cleanup model GPU memory
        for model in models.values():
            if hasattr(model, 'cleanup_gpu_memory'):
                model.cleanup_gpu_memory()
        
        # Cleanup global GPU memory
        cleanup_gpu_memory()
        
        return {"message": "GPU memory cleaned up successfully"}
        
    except Exception as e:
        logger.error(f"GPU cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unload_model/{dataset}")
async def unload_model(dataset: str):
    """Unload a specific model to free memory"""
    try:
        if dataset in models:
            # Cleanup GPU memory first
            if hasattr(models[dataset], 'cleanup_gpu_memory'):
                models[dataset].cleanup_gpu_memory()
            
            # Remove from models dict
            del models[dataset]
            
            # Force garbage collection
            gc.collect()
            
            return {"message": f"Model {dataset} unloaded successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {dataset} not found")
            
    except Exception as e:
        logger.error(f"Model unload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Simple embedding model cache
simple_model_cache = {}

def get_simple_embedding_model():
    """Get a simple SentenceTransformer model for embedding generation"""
    global simple_model_cache
    
    if "bert_model" not in simple_model_cache:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            simple_model_cache["bert_model"] = model
            logger.info("✅ Simple BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load simple BERT model: {e}")
            raise
    
    return simple_model_cache["bert_model"]

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Get embedding vector for text (for vector store service)"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        start_time = time.time()
        
        # Use simple model approach
        model = get_simple_embedding_model()
        
        # Generate embedding
        embedding = model.encode([request.text])
        if len(embedding.shape) > 1:
            embedding = embedding[0]  # Get first result
        
        # Normalize for cosine similarity
        embedding_norm = embedding / np.linalg.norm(embedding)
        
        processing_time = time.time() - start_time
        
        # Convert to list
        embedding_list = embedding_norm.tolist()
        
        return EmbedResponse(
            embedding=embedding_list,
            text=request.text,
            dataset=request.dataset,
            dimension=len(embedding_list),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "loaded_models": list(models.keys()),
        "gpu_available": gpu_accelerator.gpu_available,
        "faiss_available": gpu_accelerator.faiss_available,
        "faiss_gpu_available": gpu_accelerator.faiss_gpu_available
    }

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Starting Optimized Embedding Service...")
    logger.info(f"GPU Available: {gpu_accelerator.gpu_available}")
    logger.info(f"FAISS Available: {gpu_accelerator.faiss_available}")
    logger.info(f"FAISS GPU Available: {gpu_accelerator.faiss_gpu_available}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Optimized Embedding Service...")
    
    # Cleanup all models
    for model in models.values():
        if hasattr(model, 'cleanup_gpu_memory'):
            model.cleanup_gpu_memory()
    
    # Cleanup global GPU memory
    cleanup_gpu_memory()
    
    logger.info("Service shutdown complete")

if __name__ == "__main__":
    logger.info("Starting Optimized Embedding Service...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        workers=1,  # Single worker for GPU memory management
        access_log=True
    )


