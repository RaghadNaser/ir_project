import numpy as np
import faiss
import joblib
import os
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VectorStoreConfig:
    """Configuration for vector store building"""
    # FAISS Index Types - Choose based on dataset size and requirements
    index_type: str = "IVF"  # Options: "Flat", "IVF", "HNSW", "PQ"
    
    # IVF parameters (for large datasets)
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Search parameter for IVF
    
    # HNSW parameters (for very fast search)
    M: int = 16  # Number of connections for HNSW
    efConstruction: int = 200  # Construction parameter for HNSW
    efSearch: int = 50  # Search parameter for HNSW
    
    # PQ parameters (for memory efficiency)
    m: int = 8  # Number of subquantizers for PQ
    nbits: int = 8  # Bits per subquantizer
    
    # Memory optimization
    use_gpu: bool = False  # Enable GPU acceleration if available
    batch_size: int = 10000  # Batch size for processing
    
    # Normalization
    normalize_embeddings: bool = True  # For cosine similarity

class OptimizedVectorStore:
    """Optimized vector store for embeddings with multiple FAISS index types"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.index = None
        self.doc_mapping = None
        self.embeddings = None
        self.dimension = None
        
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity with optimized computation"""
        if not self.config.normalize_embeddings:
            return embeddings
            
        logger.info("Normalizing embeddings for cosine similarity...")
        
        # Optimized normalization using numpy
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms
        
        # Verify normalization
        sample_norm = np.linalg.norm(normalized[0])
        logger.info(f"Sample embedding norm after normalization: {sample_norm:.6f}")
        
        return normalized.astype(np.float32)
    
    def _create_faiss_index(self, dimension: int) -> faiss.Index:
        """Create optimized FAISS index based on configuration"""
        logger.info(f"Creating {self.config.index_type} index with dimension {dimension}")
        
        if self.config.index_type == "Flat":
            # Most accurate, good for small datasets
            index = faiss.IndexFlatIP(dimension)
            logger.info("Created IndexFlatIP for exact cosine similarity")
            
        elif self.config.index_type == "IVF":
            # Good balance of speed and accuracy for medium-large datasets
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
            logger.info(f"Created IndexIVFFlat with {self.config.nlist} clusters")
            
        elif self.config.index_type == "HNSW":
            # Fastest search, good for large datasets
            index = faiss.IndexHNSWFlat(dimension, self.config.M)
            index.hnsw.efConstruction = self.config.efConstruction
            logger.info(f"Created IndexHNSWFlat with M={self.config.M}")
            
        elif self.config.index_type == "PQ":
            # Most memory efficient, good for very large datasets
            index = faiss.IndexPQ(dimension, self.config.m, self.config.nbits)
            logger.info(f"Created IndexPQ with m={self.config.m}, nbits={self.config.nbits}")
            
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")
        
        # GPU acceleration if available and requested
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving index to GPU...")
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
            
        return index
    
    def _add_embeddings_in_batches(self, embeddings: np.ndarray):
        """Add embeddings to index in batches for memory efficiency"""
        n_embeddings = embeddings.shape[0]
        batch_size = self.config.batch_size
        
        logger.info(f"Adding {n_embeddings} embeddings in batches of {batch_size}")
        
        for i in range(0, n_embeddings, batch_size):
            end_idx = min(i + batch_size, n_embeddings)
            batch = embeddings[i:end_idx]
            
            self.index.add(batch)
            
            if i % (batch_size * 10) == 0:  # Log every 10 batches
                logger.info(f"Added {end_idx}/{n_embeddings} embeddings ({end_idx/n_embeddings*100:.1f}%)")
                
        logger.info(f"Successfully added all {n_embeddings} embeddings to index")
    
    def build_index(self, embeddings_path: str, doc_mapping_path: str) -> Dict:
        """Build optimized vector store index"""
        start_time = time.time()
        
        # Load embeddings
        logger.info(f"Loading embeddings from {embeddings_path}")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            
        embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
        
        # Load document mapping
        if os.path.exists(doc_mapping_path):
            doc_mapping = joblib.load(doc_mapping_path)
            logger.info(f"Loaded document mapping with {len(doc_mapping)} entries")
        else:
            logger.warning("Document mapping file not found, creating sequential mapping")
            doc_mapping = {i: f"doc_{i}" for i in range(embeddings.shape[0])}
        
        # Normalize embeddings
        normalized_embeddings = self._normalize_embeddings(embeddings)
        
        # Create FAISS index
        self.dimension = normalized_embeddings.shape[1]
        self.index = self._create_faiss_index(self.dimension)
        
        # Train index if needed (for IVF)
        if self.config.index_type == "IVF":
            logger.info("Training IVF index...")
            # Use subset for training to save memory
            train_size = min(100000, embeddings.shape[0])
            train_embeddings = normalized_embeddings[:train_size]
            self.index.train(train_embeddings)
            logger.info("IVF index training completed")
        
        # Add embeddings to index
        self._add_embeddings_in_batches(normalized_embeddings)
        
        # Set search parameters
        if self.config.index_type == "IVF":
            self.index.nprobe = self.config.nprobe
        elif self.config.index_type == "HNSW":
            self.index.hnsw.efSearch = self.config.efSearch
        
        # Store references
        self.embeddings = normalized_embeddings
        self.doc_mapping = doc_mapping
        
        # Clean up memory
        del embeddings  # Remove original embeddings
        gc.collect()
        
        build_time = time.time() - start_time
        
        # Performance metrics
        metrics = {
            "build_time": build_time,
            "n_embeddings": normalized_embeddings.shape[0],
            "dimension": self.dimension,
            "index_type": self.config.index_type,
            "memory_usage_mb": self.get_memory_usage(),
            "embeddings_per_second": normalized_embeddings.shape[0] / build_time
        }
        
        logger.info(f"Vector store built in {build_time:.2f}s")
        logger.info(f"Performance: {metrics['embeddings_per_second']:.0f} embeddings/second")
        
        return metrics
    
    def save_index(self, index_path: str, metadata_path: str):
        """Save the vector store index and metadata"""
        # Save FAISS index
        if self.config.use_gpu:
            # Move back to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            "config": self.config,
            "doc_mapping": self.doc_mapping,
            "dimension": self.dimension,
            "n_embeddings": self.index.ntotal
        }
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Index saved to {index_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def get_memory_usage(self) -> float:
        """Get approximate memory usage in MB"""
        if self.index is None:
            return 0
        
        # Estimate based on number of vectors and dimension
        n_vectors = self.index.ntotal
        bytes_per_vector = self.dimension * 4  # float32
        
        if self.config.index_type == "Flat":
            memory_mb = (n_vectors * bytes_per_vector) / (1024 * 1024)
        elif self.config.index_type == "IVF":
            # IVF uses additional memory for cluster centroids
            memory_mb = (n_vectors * bytes_per_vector * 1.1) / (1024 * 1024)
        elif self.config.index_type == "HNSW":
            # HNSW uses additional memory for graph structure
            memory_mb = (n_vectors * bytes_per_vector * 1.5) / (1024 * 1024)
        elif self.config.index_type == "PQ":
            # PQ compresses vectors significantly
            memory_mb = (n_vectors * self.config.m * self.config.nbits / 8) / (1024 * 1024)
        else:
            memory_mb = (n_vectors * bytes_per_vector) / (1024 * 1024)
        
        return memory_mb

def build_optimized_vector_store(dataset: str = "argsme", index_type: str = "IVF") -> Dict:
    """Build optimized vector store for a specific dataset"""
    
    # Configure paths
    if dataset.lower() == "argsme":
        embeddings_path = "data/vectors/argsme/embedding/argsme_bert_embeddings.npy"
        doc_mapping_path = "data/vectors/argsme/embedding/argsme_bert_doc_mapping.joblib"
        index_path = f"data/vectors/argsme/embedding/faiss_index_{index_type.lower()}.bin"
        metadata_path = f"data/vectors/argsme/embedding/faiss_metadata_{index_type.lower()}.joblib"
    elif dataset.lower() == "wikir":
        embeddings_path = "data/vectors/wikir/embedding/wikir_bert_embeddings.npy"
        doc_mapping_path = "data/vectors/wikir/embedding/wikir_bert_doc_mapping.joblib"
        index_path = f"data/vectors/wikir/embedding/faiss_index_{index_type.lower()}.bin"
        metadata_path = f"data/vectors/wikir/embedding/faiss_metadata_{index_type.lower()}.joblib"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Auto-configure based on dataset size
    if os.path.exists(embeddings_path):
        embeddings = np.load(embeddings_path)
        n_embeddings = embeddings.shape[0]
        del embeddings  # Free memory
        
        # Automatic configuration based on dataset size
        if n_embeddings < 10000:
            # Small dataset - use exact search
            config = VectorStoreConfig(index_type="Flat")
        elif n_embeddings < 100000:
            # Medium dataset - use IVF
            config = VectorStoreConfig(
                index_type="IVF",
                nlist=min(100, n_embeddings // 100),
                nprobe=10
            )
        elif n_embeddings < 1000000:
            # Large dataset - use HNSW for speed
            config = VectorStoreConfig(
                index_type="HNSW",
                M=16,
                efConstruction=200,
                efSearch=50
            )
        else:
            # Very large dataset - use PQ for memory efficiency
            config = VectorStoreConfig(
                index_type="PQ",
                m=8,
                nbits=8
            )
        
        # Override with user choice
        if index_type.upper() in ["FLAT", "IVF", "HNSW", "PQ"]:
            config.index_type = index_type.upper()
        
        logger.info(f"Auto-configured for {n_embeddings} embeddings using {config.index_type}")
else:
        # Default configuration
        config = VectorStoreConfig(index_type=index_type.upper())
    
    # Build vector store
    vector_store = OptimizedVectorStore(config)
    metrics = vector_store.build_index(embeddings_path, doc_mapping_path)
    
    # Save index
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    vector_store.save_index(index_path, metadata_path)
    
    # Final metrics
    final_metrics = {
        **metrics,
        "dataset": dataset,
        "index_path": index_path,
        "metadata_path": metadata_path
    }
    
    logger.info("=== VECTOR STORE BUILD COMPLETE ===")
    logger.info(f"Dataset: {dataset.upper()}")
    logger.info(f"Index Type: {config.index_type}")
    logger.info(f"Embeddings: {metrics['n_embeddings']:,}")
    logger.info(f"Dimension: {metrics['dimension']}")
    logger.info(f"Build Time: {metrics['build_time']:.2f}s")
    logger.info(f"Speed: {metrics['embeddings_per_second']:.0f} embeddings/second")
    logger.info(f"Memory Usage: {metrics['memory_usage_mb']:.1f} MB")
    
    return final_metrics

if __name__ == "__main__":
    # Build vector stores for both datasets with different index types
    
    # ARGSME with IVF (balanced speed/accuracy)
    print("=" * 60)
    print("BUILDING ARGSME VECTOR STORE (IVF)")
    print("=" * 60)
    argsme_metrics = build_optimized_vector_store("argsme", "IVF")
    
    # ARGSME with HNSW (fastest search)
    print("\n" + "=" * 60)
    print("BUILDING ARGSME VECTOR STORE (HNSW)")
    print("=" * 60)
    argsme_hnsw_metrics = build_optimized_vector_store("argsme", "HNSW")
    
    # WikiIR with appropriate index type
    print("\n" + "=" * 60)
    print("BUILDING WIKIR VECTOR STORE (AUTO-CONFIGURED)")
    print("=" * 60)
    try:
        wikir_metrics = build_optimized_vector_store("wikir", "IVF")
    except Exception as e:
        logger.error(f"WikiIR build failed: {e}")
        logger.info("Trying with PQ index for memory efficiency...")
        wikir_metrics = build_optimized_vector_store("wikir", "PQ")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"ARGSME IVF: {argsme_metrics['embeddings_per_second']:.0f} emb/sec, {argsme_metrics['memory_usage_mb']:.1f} MB")
    print(f"ARGSME HNSW: {argsme_hnsw_metrics['embeddings_per_second']:.0f} emb/sec, {argsme_hnsw_metrics['memory_usage_mb']:.1f} MB")
    print(f"WikiIR: {wikir_metrics['embeddings_per_second']:.0f} emb/sec, {wikir_metrics['memory_usage_mb']:.1f} MB") 