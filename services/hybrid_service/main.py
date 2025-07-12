# services/hybrid_service/main_optimized.py
"""
Optimized Hybrid Search Service
Implements Sequential and Parallel hybrid approaches for better performance
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import asyncio
import httpx
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.logging_config import setup_logging

logger = setup_logging("optimized_hybrid_service")

app = FastAPI(
    title="Optimized Hybrid Search Service",
    description="Sequential and Parallel hybrid search with performance optimization",
    version="3.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    dataset: str
    query: str
    top_k: int = 10
    method: str = "sequential"  # "sequential", "parallel", "fusion"
    first_stage_k: int = 50  # Number of candidates from first stage (sequential)
    tfidf_weight: float = 0.4
    embedding_weight: float = 0.6
    use_fast_tfidf: bool = True  # Use DTM matching for speed
    use_faiss: bool = True  # Use FAISS for embeddings


class SearchResponse(BaseModel):
    results: List[Dict]
    query: str
    dataset: str
    method: str
    execution_time: float
    total_results: int
    performance_stats: Dict


class ComparisonRequest(BaseModel):
    dataset: str
    query: str
    top_k: int = 10
    compare_methods: List[str] = ["sequential", "parallel", "fusion"]


# Service endpoints
TFIDF_SERVICE_URL = "http://localhost:8003"
EMBEDDING_SERVICE_URL = "http://localhost:8004"

# Performance stats
performance_stats = {
    "total_queries": 0,
    "sequential_queries": 0,
    "parallel_queries": 0,
    "fusion_queries": 0,
    "avg_execution_time": 0.0,
    "avg_speedup": 0.0,
}


async def call_tfidf_service(
    dataset: str, query: str, top_k: int, use_dtm: bool = True
) -> Tuple[List, float, Dict]:
    """Call TF-IDF service for fast initial filtering"""
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "dataset": dataset,
                "query": query,
                "top_k": top_k,
                "use_dtm_matching": use_dtm,
            }

            start_time = time.time()
            response = await client.post(
                f"{TFIDF_SERVICE_URL}/search", json=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                exec_time = time.time() - start_time

                # Convert to standard format with rank information
                results = []
                for rank, item in enumerate(result.get("results", [])):
                    results.append({
                        'doc_id': item["doc_id"],
                        'score': item["score"],
                        'tfidf_score': item["score"],  # Add tfidf_score
                        'tfidf_rank': rank + 1,  # 1-based ranking
                        'sources': ['tfidf']
                    })

                stats = {
                    "method": result.get("search_method", "unknown"),
                    "candidates_checked": result.get("candidates_checked", 0),
                    "cache_hit": result.get("cache_hit", False),
                }

                return results, exec_time, stats
            else:
                logger.error(f"TF-IDF service error: {response.status_code}")
                return [], 0.0, {"error": f"HTTP {response.status_code}"}

    except Exception as e:
        logger.error(f"TF-IDF service call failed: {e}")
        return [], 0.0, {"error": str(e)}


async def call_embedding_service(
    dataset: str,
    query: str,
    top_k: int,
    candidate_docs: Optional[List[str]] = None,
    use_faiss: bool = True,
) -> Tuple[List, float, Dict]:
    """Call Embedding service for semantic ranking"""
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "dataset": dataset,
                "query": query,
                "top_k": top_k,
                "use_hybrid": bool(
                    candidate_docs
                ),  # Use term filtering if candidates provided
                "use_faiss": use_faiss,
            }

            # Add candidates if provided (for sequential approach)
            if candidate_docs:
                # Note: This assumes embedding service supports candidate filtering
                # If not, we'll get full results and filter afterwards
                pass

            start_time = time.time()
            response = await client.post(
                f"{EMBEDDING_SERVICE_URL}/search", json=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                exec_time = time.time() - start_time

                # Convert to standard format with rank information
                raw_results = result.get("results", [])
                
                # Convert to detailed format
                results = []
                for rank, item in enumerate(raw_results):
                    # Handle different result formats
                    if isinstance(item, dict):
                        doc_id = item.get("doc_id")
                        score = item.get("score")
                    else:
                        doc_id, score = item
                    
                    results.append({
                        'doc_id': doc_id,
                        'score': score,
                        'embedding_score': score,  # Add embedding_score
                        'embedding_rank': rank + 1,  # 1-based ranking
                        'sources': ['embedding']
                    })

                # Filter by candidates if provided and service doesn't support it natively
                if candidate_docs and results:
                    candidate_set = set(candidate_docs)
                    filtered_results = [
                        item for item in results
                        if item['doc_id'] in candidate_set
                    ]
                    results = filtered_results[:top_k]
                else:
                    results = results[:top_k]

                stats = {
                    "method": result.get("method", "unknown"),
                    "candidates_checked": result.get("candidates_checked", 0),
                    "gpu_used": result.get("gpu_used", False),
                    "hybrid_stats": result.get("hybrid_stats", {}),
                }

                return results, exec_time, stats
            else:
                logger.error(f"Embedding service error: {response.status_code}")
                return [], 0.0, {"error": f"HTTP {response.status_code}"}

    except Exception as e:
        logger.error(f"Embedding service call failed: {e}")
        return [], 0.0, {"error": str(e)}


def rank_fusion(
    tfidf_results: List[Dict],
    embedding_results: List[Dict],
    tfidf_weight: float = 0.4,
    embedding_weight: float = 0.6,
    method: str = "weighted_sum",
) -> List[Dict]:
    """
    Fuse results from TF-IDF and Embedding models
    Methods:
    - weighted_sum: مجموع موزون للدرجات
    - rrf: Reciprocal Rank Fusion
    - borda: Borda count method
    """
    # Build lookup for fast access
    tfidf_lookup = {item['doc_id']: item for item in tfidf_results}
    emb_lookup = {item['doc_id']: item for item in embedding_results}
    all_doc_ids = set(tfidf_lookup.keys()) | set(emb_lookup.keys())

    if method == "weighted_sum":
        doc_info = {}
        for doc_id in all_doc_ids:
            tfidf_score = tfidf_lookup.get(doc_id, {}).get('score')
            emb_score = emb_lookup.get(doc_id, {}).get('score')
            tfidf_rank = tfidf_lookup.get(doc_id, {}).get('tfidf_rank')
            emb_rank = emb_lookup.get(doc_id, {}).get('embedding_rank')
            score = 0.0
            sources = []
            if tfidf_score is not None:
                score += tfidf_weight * tfidf_score
                sources.append('tfidf')
            if emb_score is not None:
                score += embedding_weight * emb_score
                sources.append('embedding')
            doc_info[doc_id] = {
                'final_score': score,
                'tfidf_score': tfidf_score,
                'tfidf_rank': tfidf_rank,
                'embedding_score': emb_score,
                'embedding_rank': emb_rank,
                'sources': sources
            }
        fused_results = [
            {
                'doc_id': doc_id,
                'score': info['final_score'],
                'tfidf_score': info['tfidf_score'],
                'tfidf_rank': info['tfidf_rank'],
                'embedding_score': info['embedding_score'],
                'embedding_rank': info['embedding_rank'],
                'sources': info['sources'],
                'fusion_method': 'weighted_sum'
            }
            for doc_id, info in sorted(doc_info.items(), key=lambda x: x[1]['final_score'], reverse=True)
        ]
        print(f'[DEBUG] Weighted_sum fusion returned {len(fused_results)} docs: {[r["doc_id"] for r in fused_results[:5]]}')
        return fused_results

    elif method == "rrf":
        doc_info = {}
        k = 60  # RRF parameter
        for doc_id in all_doc_ids:
            tfidf_rank = tfidf_lookup.get(doc_id, {}).get('tfidf_rank')
            emb_rank = emb_lookup.get(doc_id, {}).get('embedding_rank')
            tfidf_rrf = tfidf_weight / (k + (tfidf_rank if tfidf_rank is not None and tfidf_rank > 0 else k))
            emb_rrf = embedding_weight / (k + (emb_rank if emb_rank is not None and emb_rank > 0 else k))
            score = tfidf_rrf + emb_rrf
            doc_info[doc_id] = {
                'final_score': score,
                'tfidf_score': tfidf_lookup.get(doc_id, {}).get('score'),
                'tfidf_rank': tfidf_rank,
                'embedding_score': emb_lookup.get(doc_id, {}).get('score'),
                'embedding_rank': emb_rank,
                'sources': [s for s in ['tfidf' if tfidf_rank is not None else None, 'embedding' if emb_rank is not None else None] if s]
            }
        fused_results = [
            {
                'doc_id': doc_id,
                'score': info['final_score'],
                'tfidf_score': info['tfidf_score'],
                'tfidf_rank': info['tfidf_rank'],
                'embedding_score': info['embedding_score'],
                'embedding_rank': info['embedding_rank'],
                'sources': info['sources'],
                'fusion_method': 'rrf'
            }
            for doc_id, info in sorted(doc_info.items(), key=lambda x: x[1]['final_score'], reverse=True)
        ]
        print(f'[DEBUG] RRF fusion returned {len(fused_results)} docs: {[r["doc_id"] for r in fused_results[:5]]}')
        return fused_results

    elif method == "borda":
        doc_info = {}
        max_tfidf_rank = max([item['tfidf_rank'] for item in tfidf_results if item.get('tfidf_rank') is not None] or [0])
        max_emb_rank = max([item['embedding_rank'] for item in embedding_results if item.get('embedding_rank') is not None] or [0])
        max_rank = max(max_tfidf_rank, max_emb_rank, 1)
        for doc_id in all_doc_ids:
            tfidf_rank = tfidf_lookup.get(doc_id, {}).get('tfidf_rank')
            emb_rank = emb_lookup.get(doc_id, {}).get('embedding_rank')
            tfidf_borda = tfidf_weight * (max_rank - (tfidf_rank - 1) if tfidf_rank is not None and tfidf_rank > 0 else 0)
            emb_borda = embedding_weight * (max_rank - (emb_rank - 1) if emb_rank is not None and emb_rank > 0 else 0)
            score = tfidf_borda + emb_borda
            doc_info[doc_id] = {
                'final_score': score,
                'tfidf_score': tfidf_lookup.get(doc_id, {}).get('score'),
                'tfidf_rank': tfidf_rank,
                'embedding_score': emb_lookup.get(doc_id, {}).get('score'),
                'embedding_rank': emb_rank,
                'sources': [s for s in ['tfidf' if tfidf_rank is not None else None, 'embedding' if emb_rank is not None else None] if s]
            }
        fused_results = [
            {
                'doc_id': doc_id,
                'score': info['final_score'],
                'tfidf_score': info['tfidf_score'],
                'tfidf_rank': info['tfidf_rank'],
                'embedding_score': info['embedding_score'],
                'embedding_rank': info['embedding_rank'],
                'sources': info['sources'],
                'fusion_method': 'borda'
            }
            for doc_id, info in sorted(doc_info.items(), key=lambda x: x[1]['final_score'], reverse=True)
        ]
        print(f'[DEBUG] Borda fusion returned {len(fused_results)} docs: {[r["doc_id"] for r in fused_results[:5]]}')
        return fused_results

    else:
        # Default: simple concatenation
        seen = set()
        results = []
        for item in tfidf_results:
            doc_id = item['doc_id']
            if doc_id not in seen:
                results.append({
                    'doc_id': doc_id,
                    'score': item['score'],
                    'tfidf_score': item['score'],
                    'tfidf_rank': item['tfidf_rank'],
                    'embedding_score': None,
                    'embedding_rank': None,
                    'sources': ['tfidf'],
                    'fusion_method': 'concatenation'
                })
                seen.add(doc_id)
        for item in embedding_results:
            doc_id = item['doc_id']
            if doc_id not in seen:
                results.append({
                    'doc_id': doc_id,
                    'score': item['score'],
                    'tfidf_score': None,
                    'tfidf_rank': None,
                    'embedding_score': item['score'],
                    'embedding_rank': item['embedding_rank'],
                    'sources': ['embedding'],
                    'fusion_method': 'concatenation'
                })
                seen.add(doc_id)
        return results


async def sequential_hybrid_search(
    dataset: str,
    query: str,
    top_k: int = 10,
    first_stage_k: int = 50,
    use_fast_tfidf: bool = True,
    use_faiss: bool = True,
) -> Tuple[List, Dict]:
    """
    Sequential Hybrid Search (النهج التسلسلي):
    1. TF-IDF للحصول على المرشحين السريع
    2. Embeddings لترتيب المرشحين دلالياً
    """
    logger.info(
        f"Sequential search: {query} | Stage 1: {first_stage_k} → Stage 2: {top_k}"
    )

    performance_stats["sequential_queries"] += 1
    stats = {
        "stage1_time": 0.0,
        "stage2_time": 0.0,
        "total_candidates": 0,
        "filtered_candidates": 0,
        "stage1_method": "",
        "stage2_method": "",
    }

    # Stage 1: TF-IDF for initial candidate selection
    stage1_start = time.time()
    tfidf_results, stage1_time, tfidf_stats = await call_tfidf_service(
        dataset, query, first_stage_k, use_fast_tfidf
    )
    stats["stage1_time"] = stage1_time
    stats["stage1_method"] = tfidf_stats.get("method", "unknown")
    stats["total_candidates"] = tfidf_stats.get("candidates_checked", 0)

    if not tfidf_results:
        logger.warning(
            "No results from TF-IDF stage, falling back to full embedding search"
        )
        # Fallback to full embedding search
        embedding_results, stage2_time, embedding_stats = await call_embedding_service(
            dataset, query, top_k, None, use_faiss
        )
        stats["stage2_time"] = stage2_time
        stats["stage2_method"] = embedding_stats.get("method", "unknown")
        return embedding_results, stats

    # Stage 2: Embeddings on selected candidates only
    candidate_docs = [item['doc_id'] for item in tfidf_results]
    stats["filtered_candidates"] = len(candidate_docs)

    logger.info(f"Stage 2: Ranking {len(candidate_docs)} candidates with embeddings")

    # Create TF-IDF lookup for preserving original information
    tfidf_lookup = {item['doc_id']: item for item in tfidf_results}

    embedding_results, stage2_time, embedding_stats = await call_embedding_service(
        dataset, query, top_k, candidate_docs, use_faiss
    )
    stats["stage2_time"] = stage2_time
    stats["stage2_method"] = embedding_stats.get("method", "unknown")

    # If stage 2 fails, return stage 1 results with detailed info
    if not embedding_results:
        logger.warning("Embedding stage failed, returning TF-IDF results with detailed info")
        final_results = []
        for i, tfidf_item in enumerate(tfidf_results[:top_k]):
            final_results.append({
                'doc_id': tfidf_item['doc_id'],
                'score': tfidf_item['score'],
                'tfidf_score': tfidf_item['score'],
                'tfidf_rank': tfidf_item['tfidf_rank'],
                'embedding_score': None,
                'embedding_rank': None,
                'sources': ['tfidf'],
                'fusion_method': 'tfidf_only'
            })
        return final_results, stats

    # Merge TF-IDF and Embedding information for sequential results
    final_results = []
    for emb_item in embedding_results[:top_k]:
        doc_id = emb_item['doc_id']
        
        # Get original TF-IDF information
        tfidf_info = tfidf_lookup.get(doc_id, {})
        
        # Create merged result
        merged_result = {
            'doc_id': doc_id,
            'score': emb_item['score'],  # Use embedding score as final score
            'tfidf_score': tfidf_info.get('score'),
            'tfidf_rank': tfidf_info.get('tfidf_rank'),
            'embedding_score': emb_item['score'],
            'embedding_rank': emb_item['embedding_rank'],
            'sources': ['tfidf', 'embedding'],
            'fusion_method': 'sequential'
        }
        
        final_results.append(merged_result)

    logger.info(f"Sequential search completed: {len(final_results)} final results with merged info")
    return final_results, stats


async def parallel_hybrid_search(
    dataset: str,
    query: str,
    top_k: int = 10,
    tfidf_weight: float = 0.4,
    embedding_weight: float = 0.6,
    use_fast_tfidf: bool = True,
    use_faiss: bool = True,
) -> Tuple[List, Dict]:
    """
    Parallel Hybrid Search (النهج التفريعي):
    1. تشغيل TF-IDF و Embeddings بالتوازي
    2. دمج النتائج باستخدام fusion
    """
    logger.info(
        f"Parallel search: {query} | TF-IDF: {tfidf_weight}, Embedding: {embedding_weight}"
    )

    performance_stats["parallel_queries"] += 1
    stats = {
        "tfidf_time": 0.0,
        "embedding_time": 0.0,
        "fusion_time": 0.0,
        "tfidf_results": 0,
        "embedding_results": 0,
        "fused_results": 0,
    }

    # Run both models in parallel
    tfidf_task = call_tfidf_service(
        dataset, query, top_k * 2, use_fast_tfidf
    )  # Get double results for fusion
    embedding_task = call_embedding_service(dataset, query, top_k * 2, None, use_faiss)

    # Wait for both results
    (tfidf_results, tfidf_time, tfidf_stats), (
        embedding_results,
        embedding_time,
        embedding_stats,
    ) = await asyncio.gather(tfidf_task, embedding_task)

    stats["tfidf_time"] = tfidf_time
    stats["embedding_time"] = embedding_time
    stats["tfidf_results"] = len(tfidf_results)
    stats["embedding_results"] = len(embedding_results)

    # Merge results
    fusion_start = time.time()

    if not tfidf_results and not embedding_results:
        logger.warning("Both services returned no results")
        return [], stats
    elif not tfidf_results:
        logger.info("TF-IDF returned no results, using embedding results only")
        final_results = embedding_results[:top_k]
    elif not embedding_results:
        logger.info("Embedding returned no results, using TF-IDF results only")
        final_results = tfidf_results[:top_k]
    else:
        # Use weighted fusion
        fusion_methods = ["weighted_sum", "rrf", "borda"]
        fusion_results = {}

        for method in fusion_methods:
            fused = rank_fusion(
                tfidf_results,
                embedding_results,
                tfidf_weight,
                embedding_weight,
                method,
            )
            fusion_results[method] = fused[: top_k]

        # Concatenate all fusion results into a single list, each with fusion_method
        results = []
        seen_doc_ids = set()
        for method in fusion_methods:
            for item in fusion_results[method]:
                # Ensure fusion_method is set correctly
                item["fusion_method"] = method
                # Avoid duplicate doc_ids across fusion methods
                if item["doc_id"] not in seen_doc_ids:
                    results.append(item)
                    seen_doc_ids.add(item["doc_id"])
        final_results = results[:top_k]
        method_stats = {
            "tfidf_time": tfidf_time,
            "embedding_time": embedding_time,
            "fusion_methods": list(fusion_methods),
            "tfidf_results": len(tfidf_results),
            "embedding_results": len(embedding_results),
        }

    stats["fusion_time"] = time.time() - fusion_start
    stats["fused_results"] = len(final_results)

    logger.info(f"Parallel search completed: {len(final_results)} fused results")
    return final_results, stats


@app.get("/")
def root():
    return {
        "service": "Optimized Hybrid Search Service",
        "version": "3.0.0",
        "description": "Sequential and Parallel hybrid search with DTM+FAISS optimization",
        "available_methods": ["sequential", "parallel", "fusion"],
        "available_datasets": ["argsme", "wikir"],
        "performance_stats": performance_stats,
    }


@app.get("/health")
async def health_check():
    """Check health of this service and dependent services"""
    health_status = {
        "hybrid_service": "healthy",
        "tfidf_service": "unknown",
        "embedding_service": "unknown",
    }

    # Check TF-IDF service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{TFIDF_SERVICE_URL}/health", timeout=5)
            if response.status_code == 200:
                health_status["tfidf_service"] = "healthy"
            else:
                health_status["tfidf_service"] = f"unhealthy ({response.status_code})"
    except Exception as e:
        health_status["tfidf_service"] = f"error: {str(e)[:50]}"

    # Check Embedding service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{EMBEDDING_SERVICE_URL}/health", timeout=5)
            if response.status_code == 200:
                health_status["embedding_service"] = "healthy"
            else:
                health_status["embedding_service"] = (
                    f"unhealthy ({response.status_code})"
                )
    except Exception as e:
        health_status["embedding_service"] = f"error: {str(e)[:50]}"

    return health_status


@app.post("/search", response_model=SearchResponse)
async def optimized_hybrid_search(req: SearchRequest):
    """
    Optimized hybrid search with multiple strategies
    """
    start_time = time.time()
    performance_stats["total_queries"] += 1

    try:
        logger.info(f"Search request: {req.method} | {req.dataset} | '{req.query}'")

        # Validate dataset
        if req.dataset not in ["argsme", "wikir"]:
            raise HTTPException(
                status_code=400, detail=f"Unsupported dataset: {req.dataset}"
            )

        # Validate method
        if req.method not in ["sequential", "parallel", "fusion"]:
            raise HTTPException(
                status_code=400, detail=f"Unsupported method: {req.method}"
            )

        # Execute search based on method
        if req.method == "sequential":
            results, method_stats = await sequential_hybrid_search(
                req.dataset,
                req.query,
                req.top_k,
                req.first_stage_k,
                req.use_fast_tfidf,
                req.use_faiss,
            )

        elif req.method == "parallel":
            results, method_stats = await parallel_hybrid_search(
                req.dataset,
                req.query,
                req.top_k,
                req.tfidf_weight,
                req.embedding_weight,
                req.use_fast_tfidf,
                req.use_faiss,
            )

        elif req.method == "fusion":
            # Fusion is an enhanced parallel approach with multiple fusion methods
            performance_stats["fusion_queries"] += 1

            # Get more results for better fusion
            tfidf_task = call_tfidf_service(
                req.dataset, req.query, req.top_k * 3, req.use_fast_tfidf
            )
            embedding_task = call_embedding_service(
                req.dataset, req.query, req.top_k * 3, None, req.use_faiss
            )

            (tfidf_results, tfidf_time, tfidf_stats), (
                embedding_results,
                embedding_time,
                embedding_stats,
            ) = await asyncio.gather(tfidf_task, embedding_task)

            # Try multiple fusion methods and pick the best
            fusion_methods = ["weighted_sum", "rrf", "borda"]
            fusion_results = {}

            for method in fusion_methods:
                fused = rank_fusion(
                    tfidf_results,
                    embedding_results,
                    req.tfidf_weight,
                    req.embedding_weight,
                    method,
                )
                fusion_results[method] = fused[: req.top_k]

            # Concatenate all fusion results into a single list, each with fusion_method
            results = []
            for method in fusion_methods:
                for item in fusion_results[method]:
                    # Ensure fusion_method is set correctly
                    item["fusion_method"] = method
                    results.append(item)

            method_stats = {
                "tfidf_time": tfidf_time,
                "embedding_time": embedding_time,
                "fusion_methods": list(fusion_methods),
                "tfidf_results": len(tfidf_results),
                "embedding_results": len(embedding_results),
            }

        # Format results with detailed information
        formatted_results = []
        for i, result in enumerate(results):
            # Handle both old tuple format and new dict format
            if isinstance(result, dict):
                formatted_results.append({
                    "doc_id": result.get("doc_id"),
                    "score": float(result.get("score", 0)),
                    "rank": i + 1,
                    "tfidf_score": result.get("tfidf_score"),
                    "tfidf_rank": result.get("tfidf_rank"),
                    "embedding_score": result.get("embedding_score"),
                    "embedding_rank": result.get("embedding_rank"),
                    "sources": result.get("sources", []),
                    "fusion_method": result.get("fusion_method")
                })
            else:
                # Legacy tuple format - this should not happen with current implementation
                doc_id, score = result
                formatted_results.append({
                    "doc_id": doc_id,
                    "score": float(score),
                    "rank": i + 1,
                    "tfidf_score": None,
                    "tfidf_rank": None,
                    "embedding_score": None,
                    "embedding_rank": None,
                    "sources": ["unknown"],
                    "fusion_method": None
                })

        execution_time = time.time() - start_time

        # Update performance stats
        performance_stats["avg_execution_time"] = (
            performance_stats["avg_execution_time"]
            * (performance_stats["total_queries"] - 1)
            + execution_time
        ) / performance_stats["total_queries"]

        logger.info(
            f"Search completed: {len(formatted_results)} results in {execution_time:.3f}s"
        )

        return SearchResponse(
            results=formatted_results,
            query=req.query,
            dataset=req.dataset,
            method=req.method,
            execution_time=execution_time,
            total_results=len(formatted_results),
            performance_stats=method_stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        execution_time = time.time() - start_time
        return SearchResponse(
            results=[],
            query=req.query,
            dataset=req.dataset,
            method=req.method,
            execution_time=execution_time,
            total_results=0,
            performance_stats={"error": str(e)},
        )


@app.post("/compare")
async def compare_methods(req: ComparisonRequest):
    """Compare different hybrid methods performance"""
    logger.info(f"Comparing methods: {req.compare_methods} for query: '{req.query}'")

    comparison_results = {}

    for method in req.compare_methods:
        try:
            search_req = SearchRequest(
                dataset=req.dataset, query=req.query, top_k=req.top_k, method=method
            )

            start_time = time.time()
            result = await optimized_hybrid_search(search_req)

            comparison_results[method] = {
                "execution_time": result.execution_time,
                "total_results": result.total_results,
                "performance_stats": result.performance_stats,
                "top_3_results": result.results[:3] if result.results else [],
            }

        except Exception as e:
            comparison_results[method] = {
                "error": str(e),
                "execution_time": 0.0,
                "total_results": 0,
            }

    return {
        "query": req.query,
        "dataset": req.dataset,
        "comparison": comparison_results,
        "fastest_method": min(
            comparison_results.keys(),
            key=lambda x: comparison_results[x].get("execution_time", float("inf")),
        ),
    }


@app.get("/stats")
def get_performance_stats():
    """Get overall performance statistics"""
    return {
        "performance": performance_stats,
        "service_info": {
            "description": "Optimized hybrid search combining TF-IDF and embeddings",
            "strategies": {
                "sequential": "TF-IDF → filter → Embeddings → rank",
                "parallel": "TF-IDF || Embeddings → fusion",
                "fusion": "Enhanced parallel with multiple fusion methods",
            },
        },
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Optimized Hybrid Service...")
    uvicorn.run(app, host="0.0.0.0", port=8005)
