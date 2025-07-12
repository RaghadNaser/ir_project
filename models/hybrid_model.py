import sys
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
import gc
import psutil
import asyncio
from threading import Lock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.logging_config import setup_logging
from .tfidf_model import TFIDFModel
from .embedding_model import EmbeddingModel

logger = setup_logging("hybrid_model")

class HybridModel:
    """
    Optimized Hybrid Search Model combining TF-IDF and Embedding models
    Supports GPU acceleration, parallel fusion, and batch processing
    """
    
    def __init__(self, dataset_name: str, tfidf_weight: float = 0.4, embedding_weight: float = 0.6):
        self.dataset_name = dataset_name
        self.tfidf_weight = tfidf_weight
        self.embedding_weight = embedding_weight
        
        # Models
        self.tfidf_model = None
        self.embedding_model = None
        
        # Performance optimization settings
        self.parallel_enabled = True
        self.max_workers = min(4, os.cpu_count() or 1)
        self.cache_enabled = True
        self.similarity_cache = {}
        self.cache_lock = Lock()
        self.max_cache_size = 1000
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load both TF-IDF and Embedding models with error handling"""
        try:
            logger.info(f"Loading optimized models for dataset: {self.dataset_name}")
            
            # Load TF-IDF model
            self.tfidf_model = TFIDFModel(self.dataset_name)
            self.tfidf_model.load_model()
            logger.info("Optimized TF-IDF model loaded successfully")
            
            # Load Embedding model
            self.embedding_model = EmbeddingModel(self.dataset_name)
            self.embedding_model.load_model()
            logger.info("Optimized Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _to_serializable(self, results):
        """Helper: convert all numpy.float32 to float"""
        serializable = []
        for item in results:
            doc_id = item[0]
            score = float(item[1])
            if len(item) > 2 and isinstance(item[2], dict):
                indiv = {k: float(v) for k, v in item[2].items()}
                serializable.append((doc_id, score, indiv))
            else:
                serializable.append((doc_id, score))
        return serializable
    
    def _get_cache_key(self, query: str, method: str, top_k: int, **kwargs) -> str:
        """Generate cache key for query results"""
        return f"{query}_{method}_{top_k}_{hash(str(kwargs))}"
    
    def _cache_results(self, cache_key: str, results: List):
        """Cache search results with LRU eviction"""
        if not self.cache_enabled:
            return
        
        with self.cache_lock:
            if len(self.similarity_cache) >= self.max_cache_size:
                # Remove oldest entry (simple FIFO for now)
                oldest_key = next(iter(self.similarity_cache))
                del self.similarity_cache[oldest_key]
            
            self.similarity_cache[cache_key] = results
    
    def _get_cached_results(self, cache_key: str) -> Optional[List]:
        """Get cached results if available"""
        if not self.cache_enabled:
            return None
        
        with self.cache_lock:
            return self.similarity_cache.get(cache_key)
    
    def search_parallel_optimized(self, query: str, top_k: int = 10, fusion_candidates_k: int = 1000, use_dtm_matching: bool = False, return_dict: bool = False) -> List:
        """
        Optimized Parallel Hybrid Search with better memory management and unified with online service logic
        """
        if self.tfidf_model is None or self.embedding_model is None:
            self.load_models()
        # Check cache first
        cache_key = self._get_cache_key(query, f"parallel_optimized_{use_dtm_matching}", top_k, fusion_candidates_k=fusion_candidates_k)
        cached_results = self._get_cached_results(cache_key)
        if cached_results is not None:
            logger.info("Returning cached results")
            return cached_results
        logger.info(f"Performing optimized parallel hybrid search for query: '{query}' (DTM={use_dtm_matching})")
        start_time = time.time()
        try:
            if self.parallel_enabled:
                # Parallel execution of both models
                with ThreadPoolExecutor(max_workers=2) as executor:
                    tfidf_future = executor.submit(self.tfidf_model.search, query, fusion_candidates_k, None, "tfidf_only", 1000, use_dtm_matching, return_dict)
                    embedding_future = executor.submit(self.embedding_model.search, query, fusion_candidates_k)
                    tfidf_results = tfidf_future.result()
                    embedding_results = embedding_future.result()
            else:
                tfidf_results = self.tfidf_model.search(query, fusion_candidates_k, None, "tfidf_only", 1000, use_dtm_matching, return_dict)
                embedding_results = self.embedding_model.search(query, fusion_candidates_k)
            # Convert tfidf_results to dict if not already
            if not return_dict:
                tfidf_dict = {doc_id: score for doc_id, score in tfidf_results}
            else:
                tfidf_dict = {item["doc_id"]: item["score"] for item in tfidf_results}
            embedding_dict = {doc_id: score for doc_id, score in embedding_results}
            # Combine candidates efficiently
            candidate_doc_ids = list(set(tfidf_dict.keys()) | set(embedding_dict.keys()))
            if not candidate_doc_ids:
                return []
            # Batch compute final scores for all candidates
            final_results = []
            for doc_id in candidate_doc_ids:
                tfidf_score = tfidf_dict.get(doc_id, 0.0)
                embedding_score = embedding_dict.get(doc_id, 0.0)
                final_score = self.tfidf_weight * tfidf_score + self.embedding_weight * embedding_score
                final_results.append({
                    "doc_id": doc_id,
                    "score": final_score,
                    "tfidf_score": tfidf_score,
                    "embedding_score": embedding_score,
                    "sources": [s for s in ["tfidf" if tfidf_score > 0 else None, "embedding" if embedding_score > 0 else None] if s],
                    "fusion_method": "weighted_sum"
                })
            # Sort and return top results as dicts
            sorted_results = sorted(final_results, key=lambda x: x["score"], reverse=True)[:top_k]
            elapsed_time = time.time() - start_time
            logger.info(f"Optimized parallel hybrid search completed in {elapsed_time:.3f}s: {len(sorted_results)} results")
            # Cache results
            self._cache_results(cache_key, sorted_results)
            return sorted_results
        except Exception as e:
            logger.error(f"Error in optimized parallel hybrid search: {e}")
            raise
    
    def search_parallel(self, query: str, top_k: int = 10, fusion_candidates_k: int = 1000, use_dtm_matching: bool = False, return_dict: bool = False) -> List:
        """
        Legacy parallel search method - redirects to optimized version
        """
        return self.search_parallel_optimized(query, top_k, fusion_candidates_k, use_dtm_matching, return_dict)
    
    def search_serial_tfidf_first_optimized(self, query: str, top_k: int = 10, first_stage_k: int = 2000) -> List[Tuple[str, float, Dict]]:
        """
        Optimized Serial Hybrid Search: TF-IDF first stage, then Embedding reranking
        """
        # Ensure models are loaded
        if self.tfidf_model is None or self.embedding_model is None:
            self.load_models()
        
        logger.info(f"Performing optimized serial hybrid search (TF-IDF first) for query: '{query}'")
        start_time = time.time()
        
        try:
            # First stage: TF-IDF filtering
            logger.info(f"First stage: TF-IDF filtering with {first_stage_k} candidates")
            tfidf_candidates = self.tfidf_model.search(query, first_stage_k)
            candidate_doc_ids = [doc_id for doc_id, _ in tfidf_candidates]
            
            if not candidate_doc_ids:
                logger.warning("No candidates found in first stage")
                return []
            
            logger.info(f"First stage completed: {len(candidate_doc_ids)} candidates")
            
            # Second stage: Batch embedding computation
            logger.info("Second stage: Batch embedding reranking")
            embedding_results = self.embedding_model.search(query, top_k=len(candidate_doc_ids), 
                                                           candidate_doc_ids=candidate_doc_ids)
            
            # Combine scores efficiently
            final_results = self._combine_scores_serial(tfidf_candidates, embedding_results, top_k)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Optimized serial hybrid search (TF-IDF first) completed in {elapsed_time:.3f}s: {len(final_results)} results")
            
            return self._to_serializable(final_results)
            
        except Exception as e:
            logger.error(f"Error in optimized serial hybrid search (TF-IDF first): {e}")
            raise
    
    def search_serial_embedding_first_optimized(self, query: str, top_k: int = 10, first_stage_k: int = 2000) -> List[Tuple[str, float, Dict]]:
        """
        Optimized Serial Hybrid Search: Embedding first stage, then TF-IDF reranking
        """
        # Ensure models are loaded
        if self.tfidf_model is None or self.embedding_model is None:
            self.load_models()
        
        logger.info(f"Performing optimized serial hybrid search (Embedding first) for query: '{query}'")
        start_time = time.time()
        
        try:
            # First stage: Embedding filtering
            logger.info(f"First stage: Embedding filtering with {first_stage_k} candidates")
            embedding_candidates = self.embedding_model.search(query, first_stage_k)
            candidate_doc_ids = [doc_id for doc_id, _ in embedding_candidates]
            
            if not candidate_doc_ids:
                logger.warning("No candidates found in first stage")
                return []
            
            logger.info(f"First stage completed: {len(candidate_doc_ids)} candidates")
            
            # Second stage: Batch TF-IDF computation
            logger.info("Second stage: Batch TF-IDF reranking")
            tfidf_results = self.tfidf_model.search(query, top_k=len(candidate_doc_ids), 
                                                   candidate_doc_ids=candidate_doc_ids)
            
            # Combine scores efficiently
            final_results = self._combine_scores_serial(embedding_candidates, tfidf_results, top_k)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Optimized serial hybrid search (Embedding first) completed in {elapsed_time:.3f}s: {len(final_results)} results")
            
            return self._to_serializable(final_results)
            
        except Exception as e:
            logger.error(f"Error in optimized serial hybrid search (Embedding first): {e}")
            raise
    
    def _combine_scores_serial(self, first_stage_results: List, second_stage_results: List, top_k: int):
        """Efficiently combine scores from serial search stages"""
        first_scores = dict(first_stage_results)
        second_scores = dict(second_stage_results)
        
        combined_scores = {}
        individual_scores = {}
        
        # Process all documents that appear in both stages
        all_doc_ids = set(first_scores.keys()) | set(second_scores.keys())
        
        for doc_id in all_doc_ids:
            tfidf_score = first_scores.get(doc_id, 0.0) if isinstance(first_stage_results[0], tuple) and len(first_stage_results) > 0 else second_scores.get(doc_id, 0.0)
            embedding_score = second_scores.get(doc_id, 0.0) if isinstance(first_stage_results[0], tuple) and len(first_stage_results) > 0 else first_scores.get(doc_id, 0.0)
            
            # Determine which is which based on the calling method
            if doc_id in first_scores and doc_id in second_scores:
                # Document appears in both stages - use actual scores
                final_score = self.tfidf_weight * tfidf_score + self.embedding_weight * embedding_score
                individual_scores[doc_id] = {'tfidf': tfidf_score, 'embedding': embedding_score}
            else:
                # Document only in one stage - assign zero to missing stage
                final_score = max(tfidf_score, embedding_score) * 0.5  # Penalty for missing stage
                individual_scores[doc_id] = {'tfidf': tfidf_score, 'embedding': embedding_score}
            
            combined_scores[doc_id] = final_score
        
        # Sort and return top results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for doc_id, final_score in sorted_results[:top_k]:
            final_results.append((doc_id, final_score, individual_scores[doc_id]))
        
        return final_results
    
    # Legacy method redirects
    def search_serial_tfidf_first(self, query: str, top_k: int = 10, first_stage_k: int = 2000) -> List[Tuple[str, float, Dict]]:
        return self.search_serial_tfidf_first_optimized(query, top_k, first_stage_k)
    
    def search_serial_embedding_first(self, query: str, top_k: int = 10, first_stage_k: int = 2000) -> List[Tuple[str, float, Dict]]:
        return self.search_serial_embedding_first_optimized(query, top_k, first_stage_k)
    
    def batch_search(self, queries: List[str], method: str = "parallel", top_k: int = 10, **kwargs) -> List[List[Tuple[str, float, Dict]]]:
        """
        Batch search for multiple queries with GPU optimization
        """
        if not queries:
            return []
        
        logger.info(f"Starting batch search for {len(queries)} queries using method: {method}")
        start_time = time.time()
        
        try:
            # Use model-specific batch processing when available
            if method == "parallel" and hasattr(self.embedding_model, 'batch_search') and hasattr(self.tfidf_model, 'batch_search'):
                # Parallel batch processing
                with ThreadPoolExecutor(max_workers=2) as executor:
                    tfidf_future = executor.submit(self.tfidf_model.batch_search, queries, top_k)
                    embedding_future = executor.submit(self.embedding_model.batch_search, queries, top_k)
                    
                    tfidf_batch_results = tfidf_future.result()
                    embedding_batch_results = embedding_future.result()
                
                # Combine results for each query
                results = []
                for i, query in enumerate(queries):
                    tfidf_results = tfidf_batch_results[i] if i < len(tfidf_batch_results) else []
                    embedding_results = embedding_batch_results[i] if i < len(embedding_batch_results) else []
                    
                    candidate_doc_ids = list(set([doc_id for doc_id, _ in tfidf_results] + 
                                               [doc_id for doc_id, _ in embedding_results]))
                    
                    if candidate_doc_ids:
                        final_results = self._compute_final_scores_batch(
                            query, candidate_doc_ids, tfidf_results, embedding_results, top_k
                        )
                        results.append(self._to_serializable(final_results))
                    else:
                        results.append([])
                
                elapsed_time = time.time() - start_time
                logger.info(f"Batch search completed in {elapsed_time:.3f}s")
                return results
            
            else:
                # Sequential processing for non-parallel methods or when batch methods unavailable
                results = []
                for query in queries:
                    query_results = self.search(query, method=method, top_k=top_k, **kwargs)
                    results.append(query_results)
                
                elapsed_time = time.time() - start_time
                logger.info(f"Sequential batch search completed in {elapsed_time:.3f}s")
                return results
                
        except Exception as e:
            logger.error(f"Error in batch search: {e}")
            return [[] for _ in queries]
    
    def search(self, query: str, method: str = "parallel", top_k: int = 10, use_dtm_matching: bool = False, return_dict: bool = False, **kwargs) -> List:
        """
        Main search method with different fusion strategies
        Args:
            query: Search query
            method: Fusion method ('parallel', 'serial_tfidf_first', 'serial_embedding_first')
            top_k: Number of results to return
            use_dtm_matching: Whether to use DTM matching in tfidf search
            return_dict: Whether to return results as dicts (for service compatibility)
            **kwargs: Additional parameters (e.g., first_stage_k for serial methods)
        Returns:
            List of dicts (if return_dict=True) or tuples
        """
        if method == "parallel":
            fusion_candidates_k = kwargs.get('fusion_candidates_k', 1000)
            return self.search_parallel_optimized(query, top_k, fusion_candidates_k, use_dtm_matching, return_dict)
        elif method == "serial_tfidf_first":
            first_stage_k = kwargs.get('first_stage_k', 2000)
            return self.search_serial_tfidf_first_optimized(query, top_k, first_stage_k)
        elif method == "serial_embedding_first":
            first_stage_k = kwargs.get('first_stage_k', 2000)
            return self.search_serial_embedding_first_optimized(query, top_k, first_stage_k)
        else:
            raise ValueError(f"Unsupported fusion method: {method}")
    
    def get_model_info(self) -> Dict:
        """Get comprehensive information about the hybrid model"""
        info = {
            'dataset': self.dataset_name,
            'tfidf_weight': self.tfidf_weight,
            'embedding_weight': self.embedding_weight,
            'tfidf_model_loaded': self.tfidf_model is not None,
            'embedding_model_loaded': self.embedding_model is not None,
            'available_methods': ['parallel', 'serial_tfidf_first', 'serial_embedding_first'],
            'gpu_acceleration': False,
            'parallel_enabled': self.parallel_enabled,
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self.similarity_cache),
            'max_workers': self.max_workers
        }
        
        # Add GPU acceleration info
        if self.tfidf_model and hasattr(self.tfidf_model, 'use_gpu'):
            info['tfidf_gpu_enabled'] = self.tfidf_model.use_gpu
        if self.embedding_model and hasattr(self.embedding_model, 'use_gpu'):
            info['embedding_gpu_enabled'] = self.embedding_model.use_gpu
            info['gpu_acceleration'] = self.embedding_model.use_gpu
        
        return info
    
    def get_memory_usage(self) -> Dict:
        """Get detailed memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = {
            'hybrid_model_memory_mb': memory_info.rss / 1024 / 1024,
            'cache_entries': len(self.similarity_cache),
            'tfidf_model_stats': {},
            'embedding_model_stats': {}
        }
        
        # Get individual model stats
        if self.tfidf_model and hasattr(self.tfidf_model, 'get_memory_usage'):
            stats['tfidf_model_stats'] = self.tfidf_model.get_memory_usage()
        
        if self.embedding_model and hasattr(self.embedding_model, 'get_memory_usage'):
            stats['embedding_model_stats'] = self.embedding_model.get_memory_usage()
        
        return stats
    
    def clear_cache(self):
        """Clear the similarity cache"""
        with self.cache_lock:
            self.similarity_cache.clear()
        logger.info("Similarity cache cleared")
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory from both models"""
        if self.tfidf_model and hasattr(self.tfidf_model, 'cleanup_gpu_memory'):
            self.tfidf_model.cleanup_gpu_memory()
        
        if self.embedding_model and hasattr(self.embedding_model, 'cleanup_gpu_memory'):
            self.embedding_model.cleanup_gpu_memory()
        
        gc.collect()
        logger.info("Hybrid model GPU memory cleaned up")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup_gpu_memory() 