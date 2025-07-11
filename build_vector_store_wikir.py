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
class WikiIRVectorStoreConfig:
    """Optimized configuration for WikiIR vector store"""
    # Index type - optimized for WikiIR's size
    index_type: str = "IVF"  # Best for WikiIR's medium-large size
    
    # IVF parameters (optimized for WikiIR)
    nlist: int = 1000  # More clusters for larger dataset
    nprobe: int = 20   # More probes for better accuracy
    
    # HNSW parameters (for fastest search)
    M: int = 32        # More connections for larger dataset
    efConstruction: int = 400  # Higher for better graph quality
    efSearch: int = 100        # Higher for better search quality
    
    # PQ parameters (for memory efficiency)
    m: int = 16        # More subquantizers for WikiIR
    nbits: int = 8     # Standard bits per subquantizer
    
    # Memory optimization for WikiIR
    use_gpu: bool = False
    batch_size: int = 5000  # Smaller batches for WikiIR
    normalize_embeddings: bool = True
    
    # WikiIR specific optimizations
    enable_preprocessing: bool = True
    memory_efficient_loading: bool = True
    save_compressed: bool = True

class WikiIRVectorStore:
    """Optimized vector store specifically for WikiIR dataset"""
    
    def __init__(self, config: WikiIRVectorStoreConfig):
        self.config = config
        self.index = None
        self.doc_mapping = None
        self.embeddings = None
        self.dimension = None
        self.dataset_stats = {}
        
    def _check_dataset_size(self, embeddings_path: str) -> Dict:
        """Analyze WikiIR dataset size and characteristics"""
        logger.info("Analyzing WikiIR dataset characteristics...")
        
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"WikiIR embeddings not found: {embeddings_path}")
        
        # Load just the header to get dimensions
        embeddings_header = np.load(embeddings_path, mmap_mode='r', allow_pickle=True)
        n_embeddings, dimension = embeddings_header.shape
        
        # Calculate memory requirements
        memory_required_gb = (n_embeddings * dimension * 4) / (1024**3)  # float32
        
        stats = {
            "n_embeddings": n_embeddings,
            "dimension": dimension,
            "memory_required_gb": memory_required_gb,
            "file_size_gb": os.path.getsize(embeddings_path) / (1024**3)
        }
        
        logger.info(f"WikiIR Dataset Analysis:")
        logger.info(f"  📊 Embeddings: {n_embeddings:,}")
        logger.info(f"  📏 Dimension: {dimension}")
        logger.info(f"  💾 Memory Required: {memory_required_gb:.2f} GB")
        logger.info(f"  📁 File Size: {stats['file_size_gb']:.2f} GB")
        
        return stats
    
    def _optimize_config_for_size(self, n_embeddings: int):
        """Auto-optimize configuration based on WikiIR dataset size"""
        logger.info(f"Auto-optimizing configuration for {n_embeddings:,} embeddings...")
        
        if n_embeddings < 50000:
            # Medium WikiIR subset
            self.config.index_type = "IVF"
            self.config.nlist = 200
            self.config.nprobe = 10
            self.config.batch_size = 10000
            
        elif n_embeddings < 200000:
            # Large WikiIR subset
            self.config.index_type = "IVF"
            self.config.nlist = 500
            self.config.nprobe = 15
            self.config.batch_size = 5000
            
        elif n_embeddings < 500000:
            # Very large WikiIR
            self.config.index_type = "HNSW"
            self.config.M = 24
            self.config.efConstruction = 300
            self.config.efSearch = 80
            self.config.batch_size = 3000
            
        else:
            # Massive WikiIR - use PQ for memory efficiency
            self.config.index_type = "PQ"
            self.config.m = 16
            self.config.nbits = 8
            self.config.batch_size = 2000
            
        logger.info(f"  🎯 Selected Index: {self.config.index_type}")
        logger.info(f"  🔧 Batch Size: {self.config.batch_size}")
    
    def _memory_efficient_normalization(self, embeddings_path: str) -> str:
        """Normalize embeddings in chunks to handle large WikiIR dataset"""
        logger.info("Performing memory-efficient normalization...")
        
        # Create normalized file path
        normalized_path = embeddings_path.replace('.npy', '_normalized.npy')
        
        if os.path.exists(normalized_path):
            logger.info(f"Found existing normalized file: {normalized_path}")
            return normalized_path
        
        # Load embeddings in memory-mapped mode
        embeddings = np.load(embeddings_path, mmap_mode='r', allow_pickle=True)
        n_embeddings, dimension = embeddings.shape
        
        # Create output array - collect all normalized chunks
        normalized_embeddings = []
        
        # Process in chunks
        chunk_size = self.config.batch_size
        logger.info(f"Processing {n_embeddings:,} embeddings in chunks of {chunk_size:,}")
        
        for i in range(0, n_embeddings, chunk_size):
            end_idx = min(i + chunk_size, n_embeddings)
            chunk = embeddings[i:end_idx].astype(np.float32)
            
            # Normalize chunk
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normalized_chunk = chunk / norms
            
            # Add to collection
            normalized_embeddings.append(normalized_chunk)
            
            if i % (chunk_size * 10) == 0:
                progress = (end_idx / n_embeddings) * 100
                logger.info(f"  Normalized {end_idx:,}/{n_embeddings:,} ({progress:.1f}%)")
        
        # Concatenate all chunks and save as proper numpy file
        logger.info("Concatenating normalized chunks...")
        all_normalized = np.concatenate(normalized_embeddings, axis=0)
        
        # Save as regular numpy file
        logger.info(f"Saving normalized embeddings to: {normalized_path}")
        np.save(normalized_path, all_normalized)
        
        # Clean up memory
        del normalized_embeddings
        del all_normalized
        
        logger.info(f"✅ Normalized embeddings saved to: {normalized_path}")
        return normalized_path
    
    def _create_optimized_faiss_index(self, dimension: int) -> faiss.Index:
        """Create FAISS index optimized for WikiIR"""
        logger.info(f"Creating {self.config.index_type} index for WikiIR (dim={dimension})")
        
        if self.config.index_type == "Flat":
            index = faiss.IndexFlatIP(dimension)
            logger.info("Created IndexFlatIP - exact search")
            
        elif self.config.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
            logger.info(f"Created IndexIVFFlat - {self.config.nlist} clusters")
            
        elif self.config.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dimension, self.config.M)
            index.hnsw.efConstruction = self.config.efConstruction
            logger.info(f"Created IndexHNSWFlat - M={self.config.M}")
            
        elif self.config.index_type == "PQ":
            index = faiss.IndexPQ(dimension, self.config.m, self.config.nbits)
            logger.info(f"Created IndexPQ - m={self.config.m}, nbits={self.config.nbits}")
            
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")
        
        return index
    
    def _add_embeddings_efficiently(self, embeddings_path: str):
        """Add embeddings to index efficiently for large WikiIR dataset"""
        logger.info("Adding embeddings to index efficiently...")
        
        # Load embeddings in memory-mapped mode
        # Check if this is a normalized file (regular numpy) or original file (pickled)
        is_normalized_file = "_normalized" in embeddings_path
        if is_normalized_file:
            # Normalized files are regular numpy arrays
            embeddings = np.load(embeddings_path, mmap_mode='r')
        else:
            # Original files might be pickled
            embeddings = np.load(embeddings_path, mmap_mode='r', allow_pickle=True)
        
        n_embeddings = embeddings.shape[0]
        batch_size = self.config.batch_size
        
        logger.info(f"Adding {n_embeddings:,} embeddings in batches of {batch_size:,}")
        
        # Track progress
        start_time = time.time()
        
        for i in range(0, n_embeddings, batch_size):
            end_idx = min(i + batch_size, n_embeddings)
            
            # Load batch
            batch = embeddings[i:end_idx].astype(np.float32)
            
            # Add to index
            self.index.add(batch)
            
            # Progress logging
            if i % (batch_size * 5) == 0:
                elapsed = time.time() - start_time
                progress = (end_idx / n_embeddings) * 100
                speed = end_idx / elapsed if elapsed > 0 else 0
                logger.info(f"  Added {end_idx:,}/{n_embeddings:,} ({progress:.1f}%) - {speed:.0f} emb/sec")
                
            # Memory management
            del batch
            if i % (batch_size * 20) == 0:
                gc.collect()
        
        total_time = time.time() - start_time
        speed = n_embeddings / total_time
        logger.info(f"✅ Added all embeddings in {total_time:.2f}s ({speed:.0f} emb/sec)")
    
    def build_wikir_index(self, embeddings_path: str, doc_mapping_path: str) -> Dict:
        """Build optimized vector store for WikiIR dataset"""
        logger.info("=" * 60)
        logger.info("🚀 BUILDING WIKIR VECTOR STORE")
        logger.info("=" * 60)
        
        build_start_time = time.time()
        
        # Step 1: Analyze dataset
        self.dataset_stats = self._check_dataset_size(embeddings_path)
        
        # Step 2: Optimize configuration
        self._optimize_config_for_size(self.dataset_stats['n_embeddings'])
        
        # Step 3: Load document mapping
        logger.info("Loading document mapping...")
        if os.path.exists(doc_mapping_path):
            self.doc_mapping = joblib.load(doc_mapping_path)
            logger.info(f"✅ Loaded {len(self.doc_mapping):,} document mappings")
        else:
            logger.warning("Document mapping not found, creating sequential mapping...")
            self.doc_mapping = {i: f"wikir_doc_{i}" for i in range(self.dataset_stats['n_embeddings'])}
        
        # Step 4: Normalize embeddings efficiently
        if self.config.normalize_embeddings:
            normalized_path = self._memory_efficient_normalization(embeddings_path)
            embeddings_path = normalized_path
        
        # Step 5: Create FAISS index
        self.dimension = self.dataset_stats['dimension']
        self.index = self._create_optimized_faiss_index(self.dimension)
        
        # Step 6: Train index if needed
        if self.config.index_type == "IVF":
            logger.info("Training IVF index for WikiIR...")
            train_start = time.time()
            
            # Use subset for training
            train_size = min(200000, self.dataset_stats['n_embeddings'])
            # Check if this is a normalized file (regular numpy) or original file (pickled)
            is_normalized_file = "_normalized" in embeddings_path
            if is_normalized_file:
                # Normalized files are regular numpy arrays
                embeddings_sample = np.load(embeddings_path, mmap_mode='r')[:train_size]
            else:
                # Original files might be pickled
                embeddings_sample = np.load(embeddings_path, mmap_mode='r', allow_pickle=True)[:train_size]
            
            self.index.train(embeddings_sample.astype(np.float32))
            
            train_time = time.time() - train_start
            logger.info(f"✅ IVF training completed in {train_time:.2f}s")
            
            del embeddings_sample
            gc.collect()
        
        # Step 7: Add embeddings to index
        self._add_embeddings_efficiently(embeddings_path)
        
        # Step 8: Configure search parameters
        if self.config.index_type == "IVF":
            self.index.nprobe = self.config.nprobe
            logger.info(f"Set nprobe = {self.config.nprobe}")
        elif self.config.index_type == "HNSW":
            self.index.hnsw.efSearch = self.config.efSearch
            logger.info(f"Set efSearch = {self.config.efSearch}")
        
        # Step 9: Calculate metrics
        total_build_time = time.time() - build_start_time
        
        metrics = {
            "dataset": "wikir",
            "build_time": total_build_time,
            "n_embeddings": self.dataset_stats['n_embeddings'],
            "dimension": self.dimension,
            "index_type": self.config.index_type,
            "embeddings_per_second": self.dataset_stats['n_embeddings'] / total_build_time,
            "memory_usage_gb": self._estimate_memory_usage(),
            "dataset_stats": self.dataset_stats
        }
        
        logger.info("=" * 60)
        logger.info("🎉 WIKIR VECTOR STORE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"📊 Embeddings: {metrics['n_embeddings']:,}")
        logger.info(f"📏 Dimension: {metrics['dimension']}")
        logger.info(f"🏗️  Index Type: {metrics['index_type']}")
        logger.info(f"⏱️  Build Time: {metrics['build_time']:.2f}s")
        logger.info(f"🚀 Speed: {metrics['embeddings_per_second']:.0f} emb/sec")
        logger.info(f"💾 Memory Usage: {metrics['memory_usage_gb']:.2f} GB")
        
        return metrics
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in GB"""
        if self.index is None:
            return 0
        
        n_vectors = self.index.ntotal
        bytes_per_vector = self.dimension * 4  # float32
        
        if self.config.index_type == "Flat":
            memory_gb = (n_vectors * bytes_per_vector) / (1024**3)
        elif self.config.index_type == "IVF":
            memory_gb = (n_vectors * bytes_per_vector * 1.1) / (1024**3)
        elif self.config.index_type == "HNSW":
            memory_gb = (n_vectors * bytes_per_vector * 1.6) / (1024**3)
        elif self.config.index_type == "PQ":
            memory_gb = (n_vectors * self.config.m * self.config.nbits / 8) / (1024**3)
        else:
            memory_gb = (n_vectors * bytes_per_vector) / (1024**3)
        
        return memory_gb
    
    def save_wikir_index(self, base_path: str = "data/vectors/wikir/embedding/"):
        """Save WikiIR vector store with comprehensive metadata"""
        os.makedirs(base_path, exist_ok=True)
        
        # Generate filenames with index type
        index_filename = f"wikir_faiss_{self.config.index_type.lower()}_index.bin"
        metadata_filename = f"wikir_faiss_{self.config.index_type.lower()}_metadata.joblib"
        
        index_path = os.path.join(base_path, index_filename)
        metadata_path = os.path.join(base_path, metadata_filename)
        
        # Save FAISS index
        logger.info(f"Saving FAISS index to: {index_path}")
        faiss.write_index(self.index, index_path)
        
        # Save comprehensive metadata
        metadata = {
            "config": self.config,
            "doc_mapping": self.doc_mapping,
            "dimension": self.dimension,
            "n_embeddings": self.index.ntotal,
            "dataset_stats": self.dataset_stats,
            "index_type": self.config.index_type,
            "creation_time": time.time(),
            "version": "1.0"
        }
        
        logger.info(f"Saving metadata to: {metadata_path}")
        joblib.dump(metadata, metadata_path)
        
        logger.info("✅ WikiIR vector store saved successfully!")
        return index_path, metadata_path

def build_wikir_vector_store(index_type: str = "auto") -> Dict:
    """Build optimized vector store for WikiIR dataset"""
    
    # Configure paths
    embeddings_path = "data/vectors/wikir/embedding/wikir_bert_embeddings.npy"
    doc_mapping_path = "data/vectors/wikir/embedding/wikir_bert_doc_mapping.joblib"
    
    # Verify files exist
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"WikiIR embeddings not found: {embeddings_path}")
    
    # Create configuration
    config = WikiIRVectorStoreConfig()
    
    # Override index type if specified
    if index_type.upper() in ["FLAT", "IVF", "HNSW", "PQ"]:
        config.index_type = index_type.upper()
    
    # Build vector store
    logger.info(f"Building WikiIR vector store with {config.index_type} index...")
    
    vector_store = WikiIRVectorStore(config)
    metrics = vector_store.build_wikir_index(embeddings_path, doc_mapping_path)
    
    # Save the index
    index_path, metadata_path = vector_store.save_wikir_index()
    
    # Add file paths to metrics
    metrics.update({
        "index_path": index_path,
        "metadata_path": metadata_path
    })
    
    return metrics

if __name__ == "__main__":
    print("🚀 WikiIR Vector Store Builder")
    print("=" * 50)
    
    try:
        # Build with auto-optimized configuration
        print("\n📊 Building WikiIR Vector Store (Auto-Optimized)")
        print("-" * 50)
        auto_metrics = build_wikir_vector_store("auto")
        
        # Build with HNSW for fastest search
        print("\n🏃 Building WikiIR Vector Store (HNSW - Fastest)")
        print("-" * 50)
        hnsw_metrics = build_wikir_vector_store("HNSW")
        
        # Build with PQ for memory efficiency
        print("\n💾 Building WikiIR Vector Store (PQ - Memory Efficient)")
        print("-" * 50)
        pq_metrics = build_wikir_vector_store("PQ")
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 WIKIR VECTOR STORE SUMMARY")
        print("=" * 60)
        print(f"Auto-Optimized: {auto_metrics['embeddings_per_second']:.0f} emb/sec, {auto_metrics['memory_usage_gb']:.2f} GB")
        print(f"HNSW (Fastest): {hnsw_metrics['embeddings_per_second']:.0f} emb/sec, {hnsw_metrics['memory_usage_gb']:.2f} GB")
        print(f"PQ (Efficient): {pq_metrics['embeddings_per_second']:.0f} emb/sec, {pq_metrics['memory_usage_gb']:.2f} GB")
        
    except Exception as e:
        logger.error(f"Failed to build WikiIR vector store: {e}")
        raise 