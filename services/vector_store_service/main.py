from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import faiss
import joblib
import os
import time
import logging
import gc
import traceback
from pathlib import Path
from dataclasses import dataclass
import psutil
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Optimized Vector Store Service",
    description="Lightning-fast FAISS-powered similarity search with optimized performance",
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

# Configuration classes for metadata compatibility - ALL POSSIBLE VARIATIONS
@dataclass
class VectorStoreConfig:
    """Configuration for ARGSME vector store (correct name)"""
    index_type: str = "IVF"
    nlist: int = 100
    nprobe: int = 10
    M: int = 16
    efConstruction: int = 200
    efSearch: int = 50
    m: int = 8
    nbits: int = 8
    use_gpu: bool = False
    batch_size: int = 10000
    normalize_embeddings: bool = True

@dataclass
class VVectorStoreConfig:
    """Configuration for ARGSME vector store (compatibility with typo)"""
    index_type: str = "IVF"
    nlist: int = 100
    nprobe: int = 10
    M: int = 16
    efConstruction: int = 200
    efSearch: int = 50
    m: int = 8
    nbits: int = 8
    use_gpu: bool = False
    batch_size: int = 10000
    normalize_embeddings: bool = True

@dataclass
class WikiIRVectorStoreConfig:
    """Configuration for WikiIR vector store (correct name)"""
    index_type: str = "IVF"
    nlist: int = 1000
    nprobe: int = 20
    M: int = 32
    efConstruction: int = 400
    efSearch: int = 100
    m: int = 16
    nbits: int = 8
    use_gpu: bool = False
    batch_size: int = 5000
    normalize_embeddings: bool = True
    enable_preprocessing: bool = True
    memory_efficient_loading: bool = True
    save_compressed: bool = True

@dataclass  
class WWikiIRVectorStoreConfig:
    """Configuration for WikiIR vector store (compatibility with typo)"""
    index_type: str = "IVF"
    nlist: int = 1000
    nprobe: int = 20
    M: int = 32
    efConstruction: int = 400
    efSearch: int = 100
    m: int = 16
    nbits: int = 8
    use_gpu: bool = False
    batch_size: int = 5000
    normalize_embeddings: bool = True
    enable_preprocessing: bool = True
    memory_efficient_loading: bool = True
    save_compressed: bool = True

class VectorSearchRequest(BaseModel):
    dataset: str
    query_vector: List[float]
    top_k: int = 10
    index_type: Optional[str] = "auto"  # "auto", "hnsw", "ivf", "pq", "flat"

class BatchVectorSearchRequest(BaseModel):
    dataset: str
    query_vectors: List[List[float]]
    top_k: int = 10
    index_type: Optional[str] = "auto"

class VectorSearchResponse(BaseModel):
    results: List[Tuple[str, float]]
    query_vector_dim: int
    dataset: str
    index_type: str
    total_results: int
    search_time: float
    performance_stats: Dict[str, Any]

class BatchVectorSearchResponse(BaseModel):
    results: List[List[Tuple[str, float]]]
    num_queries: int
    dataset: str
    index_type: str
    total_search_time: float
    avg_search_time: float
    performance_stats: Dict[str, Any]

class OptimizedVectorStoreService:
    """Highly optimized vector store service using FAISS indices"""
    
    def __init__(self):
        self.indices = {"argsme": {}, "wikir": {}}
        self.metadata = {"argsme": {}, "wikir": {}}
        self.performance_stats = {"argsme": {}, "wikir": {}}
        self.load_stats = {"total_indices": 0, "total_vectors": 0, "total_memory_mb": 0}
        self.search_stats = {"total_searches": 0, "total_search_time": 0, "avg_search_time": 0}
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent()
        }
    
    def _load_metadata_safely(self, metadata_path: str, n_vectors: int, dataset: str) -> Dict:
        """Load metadata with comprehensive error handling and fallback"""
        if not os.path.exists(metadata_path):
            logger.info(f"    📝 Creating fallback metadata (file not found)")
            return self._create_fallback_metadata(n_vectors, dataset)
        
        try:
            # Try loading with pickle
            metadata = joblib.load(metadata_path)
            logger.info(f"    ✅ Metadata loaded successfully")
            return metadata
        except Exception as e:
            logger.warning(f"    ⚠️ Metadata load failed: {e}")
            logger.info(f"    📝 Creating fallback metadata")
            return self._create_fallback_metadata(n_vectors, dataset)
    
    def _create_fallback_metadata(self, n_vectors: int, dataset: str) -> Dict:
        """Create comprehensive fallback metadata"""
        return {
            "doc_mapping": {i: f"{dataset}_doc_{i}" for i in range(n_vectors)},
            "dimension": 384,
            "n_embeddings": n_vectors,
            "index_type": "auto",
            "creation_time": time.time(),
            "fallback": True
        }
    
    def _estimate_index_memory(self, index, index_type: str, dimension: int = 384) -> float:
        """Estimate memory usage of loaded index in MB"""
        n_vectors = index.ntotal
        
        if index_type == "hnsw":
            return (n_vectors * dimension * 4 * 1.6) / (1024 * 1024)
        elif index_type == "ivf":
            return (n_vectors * dimension * 4 * 1.1) / (1024 * 1024)
        elif index_type == "pq":
            return (n_vectors * 16 * 1) / (1024 * 1024)
        else:  # flat
            return (n_vectors * dimension * 4) / (1024 * 1024)
    
    def _find_best_index(self, dataset: str) -> Optional[str]:
        """Find the best available index for a dataset based on performance priority"""
        if dataset not in self.indices or not self.indices[dataset]:
            return None
            
        priority_order = ["hnsw", "ivf", "pq", "flat"]
        
        for index_type in priority_order:
            if index_type in self.indices[dataset]:
                return index_type
        
        available = list(self.indices[dataset].keys())
        return available[0] if available else None
    
    def load_all_indices(self) -> Dict[str, Any]:
        """Load all available optimized vector store indices with progress tracking"""
        logger.info("🚀 Loading Optimized Vector Store Indices")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        datasets = ["argsme", "wikir"]
        index_types = ["hnsw", "ivf", "pq", "flat"]
        
        total_loaded = 0
        failed_loads = 0
        
        for dataset in datasets:
            logger.info(f"📊 Loading {dataset.upper()} indices...")
            dataset_loaded = 0
            
            dataset_progress = tqdm(index_types, desc=f"{dataset.upper()}", leave=False)
            
            for index_type in dataset_progress:
                dataset_progress.set_postfix({"Loading": index_type.upper()})
                
                try:
                    # Define paths based on dataset
                    if dataset == "argsme":
                        base_path = "data/vectors/argsme/embedding/"
                        if index_type == "hnsw":
                            index_path = f"{base_path}faiss_index_hnsw.bin"
                            metadata_path = f"{base_path}faiss_metadata_hnsw.joblib"
                        elif index_type == "ivf":
                            index_path = f"{base_path}faiss_index_ivf.bin"
                            metadata_path = f"{base_path}faiss_metadata_ivf.joblib"  # Use correct metadata file
                        else:
                            continue
                    else:  # wikir
                        base_path = "data/vectors/wikir/embedding/"
                        if index_type == "hnsw":
                            index_path = f"{base_path}wikir_faiss_hnsw_index.bin"
                            metadata_path = f"{base_path}wikir_faiss_hnsw_metadata.joblib"
                        elif index_type == "ivf":
                            index_path = f"{base_path}faiss_index_ivf.bin"
                            metadata_path = f"{base_path}faiss_metadata_ivf.joblib"  # Use existing metadata
                        else:
                            continue
                    
                    if not os.path.exists(index_path):
                        continue
                    
                    # Load FAISS index
                    index_load_start = time.time()
                    index = faiss.read_index(index_path)
                    index_load_time = time.time() - index_load_start
                    
                    # Load metadata safely with fallback
                    metadata = self._load_metadata_safely(metadata_path, index.ntotal, dataset)
                    
                    # Optimize index parameters
                    self._optimize_index_parameters(index, index_type)
                    
                    # Store index and metadata
                    self.indices[dataset][index_type] = index
                    self.metadata[dataset][index_type] = metadata
                    
                    # Calculate performance stats
                    memory_usage = self._estimate_index_memory(index, index_type)
                    
                    self.performance_stats[dataset][index_type] = {
                        "load_time": index_load_time,
                        "n_embeddings": index.ntotal,
                        "index_type": index_type,
                        "memory_usage_mb": memory_usage,
                        "vectors_per_second": index.ntotal / index_load_time if index_load_time > 0 else 0
                    }
                    
                    # Update global stats
                    self.load_stats["total_indices"] += 1
                    self.load_stats["total_vectors"] += index.ntotal
                    self.load_stats["total_memory_mb"] += memory_usage
                    
                    logger.info(f"    ✅ {index_type.upper()}: {index.ntotal:,} vectors, {memory_usage:.1f}MB, {index_load_time:.2f}s")
                    
                    dataset_loaded += 1
                    total_loaded += 1
                    
                except Exception as e:
                    logger.error(f"    ❌ Failed {index_type.upper()}: {e}")
                    logger.error(f"    📋 Traceback: {traceback.format_exc()}")
                    failed_loads += 1
                    continue
                
                if total_loaded % 2 == 0:
                    gc.collect()
            
            dataset_progress.close()
            
            if dataset_loaded > 0:
                best_index = self._find_best_index(dataset)
                logger.info(f"  📈 {dataset.upper()}: {dataset_loaded} indices loaded, best: {best_index.upper() if best_index else 'None'}")
            else:
                logger.warning(f"  ⚠️ {dataset.upper()}: No indices loaded")
        
        total_time = time.time() - start_time
        
        load_summary = {
            "total_loaded": total_loaded,
            "failed_loads": failed_loads,
            "total_time": total_time,
            "total_vectors": self.load_stats["total_vectors"],
            "total_memory_mb": self.load_stats["total_memory_mb"],
            "load_speed": self.load_stats["total_vectors"] / total_time if total_time > 0 else 0,
            "memory_usage": self.get_memory_usage()
        }
        
        logger.info("=" * 60)
        logger.info("🎉 Vector Store Loading Complete")
        logger.info(f"📊 Loaded: {total_loaded} indices ({failed_loads} failed)")
        logger.info(f"🔢 Total Vectors: {self.load_stats['total_vectors']:,}")
        logger.info(f"💾 Index Memory: {self.load_stats['total_memory_mb']:.1f} MB")
        logger.info(f"⏱️ Load Time: {total_time:.2f}s ({load_summary['load_speed']:.0f} vectors/sec)")
        logger.info(f"🖥️ System Memory: {load_summary['memory_usage']['rss_mb']:.1f} MB ({load_summary['memory_usage']['percent']:.1f}%)")
        logger.info("=" * 60)
        
        return load_summary
    
    def _optimize_index_parameters(self, index, index_type: str):
        """Optimize index parameters for best performance"""
        try:
            if index_type == "ivf":
                if hasattr(index, 'nprobe'):
                    index.nprobe = min(20, max(10, index.nlist // 10))
            elif index_type == "hnsw":
                if hasattr(index, 'hnsw') and hasattr(index.hnsw, 'efSearch'):
                    index.hnsw.efSearch = 100
        except Exception as e:
            logger.warning(f"Failed to optimize {index_type} parameters: {e}")
    
    def search_vector(self, query_vector: np.ndarray, dataset: str, top_k: int = 10, 
                     index_type: str = "auto") -> Dict[str, Any]:
        """Perform optimized vector similarity search"""
        try:
            search_start_time = time.time()
            
            # Validate dataset
            if dataset not in self.indices or not self.indices[dataset]:
                raise HTTPException(status_code=400, detail=f"No indices available for dataset: {dataset}")
            
            # Select index type
            if index_type == "auto":
                index_type = self._find_best_index(dataset)
                if not index_type:
                    raise HTTPException(status_code=400, detail=f"No indices available for {dataset}")
            
            if index_type not in self.indices[dataset]:
                available = list(self.indices[dataset].keys())
                raise HTTPException(status_code=400, detail=f"Index {index_type} not available. Available: {available}")
            
            # Get index and metadata
            index = self.indices[dataset][index_type]
            metadata = self.metadata[dataset][index_type]
            doc_mapping = metadata.get("doc_mapping", {})
            
            # Prepare query vector
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # Ensure correct data type and dimensions
            query_vector = query_vector.astype(np.float32)
            expected_dim = 384  # BERT dimension
            if query_vector.shape[1] != expected_dim:
                raise ValueError(f"Query vector dimension {query_vector.shape[1]} doesn't match expected {expected_dim}")
            
            # Normalize vector for cosine similarity
            norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
            query_vector_normalized = query_vector / np.where(norms == 0, 1, norms)
            query_vector_normalized = query_vector_normalized.astype(np.float32)
            
            # Perform FAISS search
            faiss_start = time.time()
            actual_top_k = min(top_k, index.ntotal)
            
            distances, indices = index.search(query_vector_normalized, actual_top_k)
            faiss_time = time.time() - faiss_start
            
            # Convert results
            results = []
            for i in range(len(distances[0])):
                doc_idx = int(indices[0][i])
                similarity = float(distances[0][i])
                
                doc_id = doc_mapping.get(doc_idx, f"{dataset}_doc_{doc_idx}")
                results.append((doc_id, similarity))
            
            total_time = time.time() - search_start_time
            
            # Update search statistics
            self.search_stats["total_searches"] += 1
            self.search_stats["total_search_time"] += total_time
            self.search_stats["avg_search_time"] = self.search_stats["total_search_time"] / self.search_stats["total_searches"]
            
            # Performance statistics
            performance_stats = {
                "total_time": total_time,
                "faiss_search_time": faiss_time,
                "preprocessing_time": search_start_time + faiss_time - faiss_start,
                "postprocessing_time": total_time - faiss_time - (search_start_time + faiss_time - faiss_start),
                "index_type": index_type,
                "n_candidates": index.ntotal,
                "results_returned": len(results),
                "search_speed_ms": total_time * 1000,
                "vectors_per_second": index.ntotal / faiss_time if faiss_time > 0 else 0,
                "throughput_qps": 1 / total_time if total_time > 0 else 0
            }
            
            return {
                "results": results,
                "query_vector_dim": query_vector.shape[1],
                "dataset": dataset,
                "index_type": index_type,
                "total_results": len(results),
                "search_time": total_time,
                "performance_stats": performance_stats
            }
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e

# Initialize service
vector_store_service = OptimizedVectorStoreService()

@app.on_event("startup")
async def startup_event():
    """Initialize the vector store service"""
    logger.info("🚀 Starting Optimized Vector Store Service")
    
    load_summary = vector_store_service.load_all_indices()
    
    if load_summary["total_loaded"] == 0:
        logger.error("❌ No vector stores loaded. Service will not work properly.")
    else:
        logger.info("✅ Optimized Vector Store Service ready!")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Optimized Vector Store Service",
        "version": "1.0.0",
        "description": "Lightning-fast FAISS-powered similarity search",
        "features": [
            "Multiple index types (HNSW, IVF, PQ, Flat)",
            "Auto-optimization",
            "Performance monitoring",
            "Memory efficient loading"
        ],
        "endpoints": {
            "/search": "POST - Single vector similarity search",
            "/health": "GET - Health check",
            "/stats": "GET - Performance statistics",
            "/indices": "GET - Available indices information"
        }
    }

@app.post("/search", response_model=VectorSearchResponse)
async def search_vector(request: VectorSearchRequest):
    """Perform optimized vector similarity search"""
    try:
        # Validate request
        if not request.query_vector:
            raise HTTPException(status_code=400, detail="Query vector cannot be empty")
        
        if request.dataset not in ["argsme", "wikir"]:
            raise HTTPException(status_code=400, detail="Dataset must be 'argsme' or 'wikir'")
        
        if request.top_k < 1 or request.top_k > 1000:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 1000")
        
        # Convert to numpy array
        query_vector = np.array(request.query_vector, dtype=np.float32)
        
        # Perform search
        results = vector_store_service.search_vector(
            query_vector=query_vector,
            dataset=request.dataset,
            top_k=request.top_k,
            index_type=request.index_type or "auto"
        )
        
        return VectorSearchResponse(**results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/batch_search", response_model=BatchVectorSearchResponse)
async def batch_search_vectors(request: BatchVectorSearchRequest):
    """Perform optimized batch vector similarity search"""
    try:
        # Validate request
        if not request.query_vectors:
            raise HTTPException(status_code=400, detail="Query vectors cannot be empty")
        
        if len(request.query_vectors) > 100:
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 queries")
        
        if request.dataset not in ["argsme", "wikir"]:
            raise HTTPException(status_code=400, detail="Dataset must be 'argsme' or 'wikir'")
        
        # Convert to numpy array
        query_vectors = np.array(request.query_vectors, dtype=np.float32)
        
        # Perform batch search
        results = vector_store_service.batch_search_vectors(
            query_vectors=query_vectors,
            dataset=request.dataset,
            top_k=request.top_k,
            index_type=request.index_type
        )
        
        return BatchVectorSearchResponse(**results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch search error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    memory_usage = vector_store_service.get_memory_usage()
    
    return {
        "status": "healthy",
        "service": "Optimized Vector Store Service",
        "indices_loaded": {
            "argsme": list(vector_store_service.indices["argsme"].keys()),
            "wikir": list(vector_store_service.indices["wikir"].keys())
        },
        "total_indices": vector_store_service.load_stats["total_indices"],
        "total_vectors": vector_store_service.load_stats["total_vectors"],
        "memory_usage": memory_usage,
        "search_stats": vector_store_service.search_stats,
        "timestamp": time.time()
    }

@app.get("/stats")
async def get_performance_stats():
    """Get detailed performance statistics"""
    return {
        "load_stats": vector_store_service.load_stats,
        "search_stats": vector_store_service.search_stats,
        "performance_stats": vector_store_service.performance_stats,
        "memory_usage": vector_store_service.get_memory_usage(),
        "available_indices": {
            "argsme": list(vector_store_service.indices["argsme"].keys()),
            "wikir": list(vector_store_service.indices["wikir"].keys())
        }
    }

@app.get("/indices")
async def get_indices_info():
    """Get detailed information about available indices"""
    info = {"datasets": {}}
    
    for dataset in ["argsme", "wikir"]:
        info["datasets"][dataset] = {}
        
        for index_type, metadata in vector_store_service.metadata[dataset].items():
            perf_stats = vector_store_service.performance_stats[dataset].get(index_type, {})
            
            info["datasets"][dataset][index_type] = {
                "n_embeddings": metadata.get("n_embeddings", 0),
                "dimension": metadata.get("dimension", 0),
                "index_type": metadata.get("index_type", index_type),
                "memory_usage_mb": perf_stats.get("memory_usage_mb", 0),
                "load_time": perf_stats.get("load_time", 0),
                "vectors_per_second": perf_stats.get("vectors_per_second", 0),
                "fallback_metadata": metadata.get("fallback", False)
            }
    
    return info

@app.get("/test/{dataset}")
async def test_search(dataset: str, dim: int = 384, top_k: int = 5):
    """Test endpoint with random vector for quick verification"""
    try:
        # Generate random test vector
        test_vector = np.random.randn(dim).astype(np.float32).tolist()
        
        request = VectorSearchRequest(
            dataset=dataset, 
            query_vector=test_vector, 
            top_k=top_k
        )
        
        return await search_vector(request)
        
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007) 