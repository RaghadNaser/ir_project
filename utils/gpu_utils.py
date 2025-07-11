"""
GPU Utilities for Information Retrieval System
Provides GPU acceleration functions for cosine similarity, FAISS indexing, and memory management
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Union
import gc

# GPU acceleration imports
try:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = None
    cp_sparse = None
    GPU_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
    # Check if FAISS GPU is available
    try:
        faiss.StandardGpuResources()
        FAISS_GPU_AVAILABLE = True
    except:
        FAISS_GPU_AVAILABLE = False
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

class GPUAccelerator:
    """Main class for GPU-accelerated operations"""
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        self.faiss_available = FAISS_AVAILABLE
        self.faiss_gpu_available = FAISS_GPU_AVAILABLE
        
        # GPU resources
        self.gpu_resource = None
        if self.faiss_gpu_available:
            try:
                self.gpu_resource = faiss.StandardGpuResources()
                logger.info("FAISS GPU resources initialized")
            except:
                self.faiss_gpu_available = False
                logger.warning("Failed to initialize FAISS GPU resources")
        
        logger.info(f"GPU Accelerator initialized - GPU: {self.gpu_available}, FAISS: {self.faiss_available}, FAISS GPU: {self.faiss_gpu_available}")
    
    def cosine_similarity_gpu(self, query_vectors: np.ndarray, doc_vectors: np.ndarray, 
                             top_k: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        GPU-accelerated cosine similarity computation
        
        Args:
            query_vectors: Query vectors (n_queries, n_features)
            doc_vectors: Document vectors (n_docs, n_features)
            top_k: If specified, return only top-k results
            
        Returns:
            Similarity scores or (scores, indices) if top_k specified
        """
        if not self.gpu_available:
            # Fallback to CPU sklearn
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vectors, doc_vectors)
            if top_k is not None:
                top_indices = np.argpartition(similarities, -top_k, axis=1)[:, -top_k:]
                # Sort within top-k
                sorted_indices = np.argsort(np.take_along_axis(similarities, top_indices, axis=1), axis=1)[:, ::-1]
                top_indices = np.take_along_axis(top_indices, sorted_indices, axis=1)
                top_similarities = np.take_along_axis(similarities, top_indices, axis=1)
                return top_similarities, top_indices
            return similarities
        
        # GPU computation
        query_gpu = cp.asarray(query_vectors)
        docs_gpu = cp.asarray(doc_vectors)
        
        # Normalize vectors
        query_norm = query_gpu / cp.linalg.norm(query_gpu, axis=1, keepdims=True)
        docs_norm = docs_gpu / cp.linalg.norm(docs_gpu, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = cp.dot(query_norm, docs_norm.T)
        
        if top_k is not None and top_k < similarities.shape[1]:
            # Get top-k efficiently
            top_indices = cp.argpartition(similarities, -top_k, axis=1)[:, -top_k:]
            top_similarities = cp.take_along_axis(similarities, top_indices, axis=1)
            
            # Sort within top-k
            sorted_indices = cp.argsort(top_similarities, axis=1)[:, ::-1]
            top_indices = cp.take_along_axis(top_indices, sorted_indices, axis=1)
            top_similarities = cp.take_along_axis(top_similarities, sorted_indices, axis=1)
            
            return top_similarities.get(), top_indices.get()
        
        return similarities.get()
    
    def create_faiss_index(self, embeddings: np.ndarray, use_gpu: bool = True, 
                          index_type: str = "flat") -> Optional[faiss.Index]:
        """
        Create a FAISS index for fast similarity search
        
        Args:
            embeddings: Document embeddings (n_docs, n_features)
            use_gpu: Whether to use GPU version
            index_type: Type of index ("flat", "ivf", "hnsw")
            
        Returns:
            FAISS index or None if failed
        """
        if not self.faiss_available:
            logger.warning("FAISS not available")
            return None
        
        n_docs, n_features = embeddings.shape
        
        try:
            if index_type == "flat":
                index = faiss.IndexFlatIP(n_features)  # Inner Product (cosine similarity for normalized vectors)
            elif index_type == "ivf":
                # IVF index for larger datasets
                n_clusters = min(int(np.sqrt(n_docs)), 1000)
                quantizer = faiss.IndexFlatIP(n_features)
                index = faiss.IndexIVFFlat(quantizer, n_features, n_clusters)
            elif index_type == "hnsw":
                # HNSW index for very fast search
                index = faiss.IndexHNSWFlat(n_features, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 100
            else:
                raise ValueError(f"Unknown index type: {index_type}")
            
            # Move to GPU if available and requested
            if use_gpu and self.faiss_gpu_available and self.gpu_resource:
                index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, index)
                logger.info(f"FAISS {index_type} index created on GPU")
            else:
                logger.info(f"FAISS {index_type} index created on CPU")
            
            # Normalize embeddings for cosine similarity
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Train index if needed (for IVF)
            if index_type == "ivf":
                if hasattr(index, 'is_trained') and not index.is_trained:
                    index.train(normalized_embeddings.astype(np.float32))
            
            # Add embeddings to index
            index.add(normalized_embeddings.astype(np.float32))
            
            logger.info(f"FAISS index built with {n_docs} vectors")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return None
    
    def search_faiss_index(self, index: faiss.Index, query_vectors: np.ndarray, 
                          top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using FAISS index
        
        Args:
            index: FAISS index
            query_vectors: Query vectors (n_queries, n_features)
            top_k: Number of results to return
            
        Returns:
            (scores, indices) arrays
        """
        if index is None:
            return np.array([]), np.array([])
        
        # Normalize query vectors
        normalized_queries = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        
        # Search
        scores, indices = index.search(normalized_queries.astype(np.float32), top_k)
        
        return scores, indices
    
    def sparse_matrix_to_gpu(self, sparse_matrix):
        """Convert scipy sparse matrix to GPU sparse matrix"""
        if not self.gpu_available or cp_sparse is None:
            return sparse_matrix
        
        try:
            if hasattr(sparse_matrix, 'tocsr'):
                sparse_matrix = sparse_matrix.tocsr()
            return cp_sparse.csr_matrix(sparse_matrix)
        except Exception as e:
            logger.warning(f"Failed to convert sparse matrix to GPU: {e}")
            return sparse_matrix
    
    def batch_process_embeddings(self, embedding_function, texts: List[str], 
                                batch_size: int = 64) -> np.ndarray:
        """
        Process embeddings in batches for better GPU utilization
        
        Args:
            embedding_function: Function to compute embeddings
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Stacked embeddings array
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embedding_function(batch_texts)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def get_memory_stats(self) -> dict:
        """Get GPU memory statistics"""
        stats = {
            'gpu_available': self.gpu_available,
            'faiss_available': self.faiss_available,
            'faiss_gpu_available': self.faiss_gpu_available
        }
        
        if self.gpu_available and cp is not None:
            try:
                memory_pool = cp.get_default_memory_pool()
                stats.update({
                    'gpu_memory_used_mb': memory_pool.used_bytes() / 1024 / 1024,
                    'gpu_memory_total_mb': memory_pool.total_bytes() / 1024 / 1024
                })
            except:
                pass
        
        return stats
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        if self.gpu_available and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            logger.info("GPU memory cleaned up")
    
    def __del__(self):
        """Cleanup resources"""
        self.cleanup_gpu_memory()

# Global accelerator instance
_gpu_accelerator = None

def get_gpu_accelerator() -> GPUAccelerator:
    """Get the global GPU accelerator instance"""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator()
    return _gpu_accelerator

# Convenience functions
def cosine_similarity_gpu(query_vectors: np.ndarray, doc_vectors: np.ndarray, 
                         top_k: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """GPU-accelerated cosine similarity - convenience function"""
    return get_gpu_accelerator().cosine_similarity_gpu(query_vectors, doc_vectors, top_k)

def create_faiss_index(embeddings: np.ndarray, use_gpu: bool = True, 
                      index_type: str = "flat") -> Optional[faiss.Index]:
    """Create FAISS index - convenience function"""
    return get_gpu_accelerator().create_faiss_index(embeddings, use_gpu, index_type)

def search_faiss_index(index: faiss.Index, query_vectors: np.ndarray, 
                      top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Search FAISS index - convenience function"""
    return get_gpu_accelerator().search_faiss_index(index, query_vectors, top_k)

def cleanup_gpu_memory():
    """Clean up GPU memory - convenience function"""
    accelerator = get_gpu_accelerator()
    accelerator.cleanup_gpu_memory()

def get_gpu_memory_stats() -> dict:
    """Get GPU memory statistics - convenience function"""
    return get_gpu_accelerator().get_memory_stats() 