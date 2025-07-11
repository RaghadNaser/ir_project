# import numpy as np
# import joblib
# from sentence_transformers import SentenceTransformer
# import re
# from pathlib import Path
# import sys
# import os
# import logging
# import psutil
# from typing import List, Tuple, Optional, Union
# import gc

# # GPU acceleration imports
# try:
#     import cupy as cp
#     import torch

#     GPU_AVAILABLE = torch.cuda.is_available()
#     DEVICE = torch.device("cuda" if GPU_AVAILABLE else "cpu")
# except ImportError:
#     cp = None
#     torch = None
#     GPU_AVAILABLE = False
#     DEVICE = "cpu"

# sys.path.append(
#     os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# )

# from config.settings import DATASETS, BERT_CONFIG
# from config.logging_config import setup_logging
# from utils.path_utils import get_embedding_paths

# logger = setup_logging("embedding_model")


# class EmbeddingModel:
#     def __init__(self, dataset_name):
#         self.dataset_name = dataset_name
#         self.model = None
#         self.embeddings = None
#         self.doc_mapping = None
#         self.reverse_mapping = None
#         self.loaded = False

#         # GPU optimization settings
#         self.use_gpu = GPU_AVAILABLE
#         self.device = DEVICE
#         self.embeddings_gpu = None
#         self.batch_size = 64 if self.use_gpu else 32

#         logger.info(f"GPU Available: {GPU_AVAILABLE}, Device: {self.device}")

#     def load_model(self):
#         try:
#             # Use dynamic paths
#             paths = get_embedding_paths(self.dataset_name)

#             # Load BERT model with GPU support
#             model_name = BERT_CONFIG["model_name"]
#             self.model = SentenceTransformer(model_name, device=str(self.device))
#             logger.info(f"BERT model loaded on device: {self.device}")

#             # Load components
#             self.embeddings = np.load(paths["embeddings"])
#             self.doc_mapping = joblib.load(paths["mapping"])

#             # Load embeddings to GPU if available
#             if self.use_gpu and cp is not None:
#                 logger.info("Loading embeddings to GPU...")
#                 self.embeddings_gpu = cp.asarray(self.embeddings)
#                 logger.info(f"Embeddings loaded to GPU: {self.embeddings_gpu.shape}")

#             # Load info if available
#             try:
#                 self.info = joblib.load(paths["info"])
#             except FileNotFoundError:
#                 self.info = {}

#             # Create reverse mapping
#             if self.dataset_name == "wikir" and isinstance(self.doc_mapping, list):
#                 # For wikir: doc_mapping is a list, index is idx, value is doc_id
#                 self.reverse_mapping = {
#                     idx: doc_id for idx, doc_id in enumerate(self.doc_mapping)
#                 }
#             else:
#                 # For other datasets: doc_mapping is a dict {doc_id: idx}
#                 self.reverse_mapping = {
#                     idx: doc_id for doc_id, idx in self.doc_mapping.items()
#                 }

#             self.loaded = True
#             logger.info(
#                 f"BERT model loaded for {self.dataset_name} with GPU acceleration: {self.use_gpu}"
#             )

#         except Exception as e:
#             logger.error(f"Error loading BERT model for {self.dataset_name}: {e}")
#             raise

#     def preprocess_query(self, query_text):
#         if not isinstance(query_text, str):
#             return ""

#         text = str(query_text).strip()

#         if self.dataset_name == "clinical":
#             text = re.sub(r"nct\\d+", "", text)
#             text = re.sub(r"\\b\\d{4}-\\d{2}-\\d{2}\\b", "", text)
#             text = re.sub(
#                 r"\\b\\d+\\s*(mg|ml|kg|cm|mm|years?|months?|days?|hours?)\\b",
#                 "DOSAGE",
#                 text,
#             )

#         text = re.sub(r"http\\S+|www\\S+", "", text)
#         text = re.sub(r"\\S+@\\S+", "", text)
#         text = re.sub(r"\\s+", " ", text)

#         return text.strip()

#     def _gpu_cosine_similarity(self, query_embedding, doc_embeddings, top_k=None):
#         """GPU-accelerated cosine similarity computation"""
#         if cp is None or not self.use_gpu:
#             # Fallback to CPU
#             from sklearn.metrics.pairwise import cosine_similarity

#             return cosine_similarity(query_embedding, doc_embeddings).flatten()

#         # Convert to CuPy arrays
#         query_gpu = cp.asarray(query_embedding)
#         docs_gpu = (
#             cp.asarray(doc_embeddings)
#             if not isinstance(doc_embeddings, cp.ndarray)
#             else doc_embeddings
#         )

#         # Normalize vectors
#         query_norm = query_gpu / cp.linalg.norm(query_gpu, axis=1, keepdims=True)
#         docs_norm = docs_gpu / cp.linalg.norm(docs_gpu, axis=1, keepdims=True)

#         # Compute cosine similarity
#         similarities = cp.dot(query_norm, docs_norm.T).flatten()

#         # If top_k is specified, get top results efficiently
#         if top_k is not None and top_k < len(similarities):
#             top_indices = cp.argpartition(similarities, -top_k)[-top_k:]
#             top_similarities = similarities[top_indices]
#             # Sort the top results
#             sorted_indices = cp.argsort(top_similarities)[::-1]
#             return (
#                 top_similarities[sorted_indices].get(),
#                 top_indices[sorted_indices].get(),
#             )

#         return similarities.get()

#     def _batch_encode_queries(self, queries: List[str]) -> np.ndarray:
#         """Batch encode multiple queries for better GPU utilization"""
#         if not isinstance(queries, list):
#             queries = [queries]

#         processed_queries = [self.preprocess_query(q) for q in queries]

#         # Use larger batch size for GPU
#         batch_size = self.batch_size if self.use_gpu else 32

#         return self.model.encode(
#             processed_queries, batch_size=batch_size, show_progress_bar=False
#         )

#     def search(
#         self, query_text, top_k=100, candidate_doc_ids=None, top_n_candidates=1000
#     ):
#         if not self.loaded:
#             self.load_model()

#         if not query_text or not query_text.strip():
#             return []

#         try:
#             # Check if model is loaded
#             if (
#                 self.model is None
#                 or self.embeddings is None
#                 or self.reverse_mapping is None
#             ):
#                 return []

#             # Preprocess and encode query
#             query_embedding = self._batch_encode_queries([query_text])

#             if candidate_doc_ids is not None:
#                 # Search within candidate documents
#                 return self._search_candidates(
#                     query_embedding, candidate_doc_ids, top_k
#                 )
#             else:
#                 # Full search with GPU acceleration
#                 return self._search_full(query_embedding, top_k, top_n_candidates)

#         except Exception as e:
#             logger.error(f"Error in BERT search: {e}")
#             return []

#     def search_faiss(self, query_text, top_k=10, candidate_doc_ids=None):
#         """
#         Search using FAISS vector store (index) built on embeddings.
#         Only for datasets with prebuilt FAISS index (e.g., argsme).
#         """
#         import faiss
#         import numpy as np
#         import os

#         # Paths
#         INDEX_PATH = "data/vectors/argsme/embedding/faiss_index_argsme.bin"
#         EMBEDDINGS_NORM_PATH = (
#             "data/vectors/argsme/embedding/argsme_bert_embeddings_norm.npy"
#         )
#         DOC_MAPPING_PATH = (
#             "data/vectors/argsme/embedding/argsme_bert_doc_mapping.joblib"
#         )

#         # Load FAISS index (load once and cache if needed)
#         if not hasattr(self, "_faiss_index") or self._faiss_index is None:
#             if not os.path.exists(INDEX_PATH):
#                 raise RuntimeError(
#                     "FAISS index file not found. Please build the vector store first."
#                 )
#             self._faiss_index = faiss.read_index(INDEX_PATH)
#             # Load normalized embeddings for dimension check (optional)
#             self._embeddings_norm = np.load(EMBEDDINGS_NORM_PATH)
#             # Load doc mapping
#             self._doc_mapping = joblib.load(DOC_MAPPING_PATH)

#         # Preprocess and encode query
#         query_proc = self.preprocess_query(query_text)
#         query_vec = self.model.encode([query_proc])
#         # Normalize
#         query_vec = query_vec / (
#             np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-10
#         )

#         # Search
#         D, I = self._faiss_index.search(query_vec.astype(np.float32), top_k)

#         # Prepare results
#         results = []
#         for idx, score in zip(I[0], D[0]):
#             # Map index to doc_id
#             if isinstance(self._doc_mapping, dict):
#                 # Reverse mapping: idx -> doc_id
#                 doc_id = [k for k, v in self._doc_mapping.items() if v == idx]
#                 doc_id = doc_id[0] if doc_id else idx
#             elif isinstance(self._doc_mapping, list):
#                 doc_id = self._doc_mapping[idx] if idx < len(self._doc_mapping) else idx
#             else:
#                 doc_id = idx
#             results.append((doc_id, float(score)))
#         return results

#     def _search_candidates(self, query_embedding, candidate_doc_ids, top_k):
#         """Search within specific candidate documents"""
#         if self.doc_mapping is not None:
#             if isinstance(self.doc_mapping, list):
#                 # For wikir: doc_id is index in list
#                 indices = [
#                     doc_id
#                     for doc_id in candidate_doc_ids
#                     if isinstance(doc_id, int) and 0 <= doc_id < len(self.doc_mapping)
#                 ]
#             else:
#                 indices = [
#                     self.doc_mapping[doc_id]
#                     for doc_id in candidate_doc_ids
#                     if doc_id in self.doc_mapping
#                 ]
#         else:
#             indices = []

#         if not indices:
#             return []

#         # Get candidate embeddings
#         if self.use_gpu and self.embeddings_gpu is not None:
#             candidate_embeddings = self.embeddings_gpu[indices]
#             similarities = self._gpu_cosine_similarity(
#                 query_embedding, candidate_embeddings
#             )
#         else:
#             candidate_embeddings = self.embeddings[indices]
#             from sklearn.metrics.pairwise import cosine_similarity

#             similarities = cosine_similarity(
#                 query_embedding, candidate_embeddings
#             ).flatten()

#         # Get top results
#         top_indices = np.argsort(similarities)[::-1][:top_k]

#         results = []
#         for idx in top_indices:
#             if similarities[idx] > 1e-10:
#                 doc_id = candidate_doc_ids[idx]
#                 if doc_id:
#                     results.append((doc_id, float(similarities[idx])))

#         return results

#     def _search_full(self, query_embedding, top_k, top_n_candidates):
#         """Full search across all embeddings"""
#         if self.use_gpu and self.embeddings_gpu is not None:
#             # GPU-accelerated search
#             result = self._gpu_cosine_similarity(
#                 query_embedding, self.embeddings_gpu, top_n_candidates
#             )
#             if isinstance(result, tuple):
#                 similarities, top_indices = result
#             else:
#                 similarities = result
#                 top_indices = np.argsort(similarities)[::-1][:top_n_candidates]
#                 similarities = similarities[top_indices]
#         else:
#             # CPU fallback
#             from sklearn.metrics.pairwise import cosine_similarity

#             similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
#             top_indices = np.argsort(similarities)[::-1][:top_n_candidates]
#             similarities = similarities[top_indices]
#         # Get document IDs for top results
#         doc_ids = [self.reverse_mapping.get(idx) for idx in top_indices]
#         # Filter and format results
#         results = []
#         final_indices = np.argsort(similarities)[::-1][:top_k]
#         # Debug: Print top similarities
#         print(f"[DEBUG] Top similarities: {similarities[final_indices]}")
#         for idx in final_indices:
#             if similarities[idx] > 1e-10:
#                 doc_id = doc_ids[idx]
#                 if doc_id:
#                     results.append((doc_id, float(similarities[idx])))
#         return results

#     def batch_search(
#         self, queries: List[str], top_k=100
#     ) -> List[List[Tuple[str, float]]]:
#         """Batch search for multiple queries - GPU optimized"""
#         if not self.loaded:
#             self.load_model()

#         if not queries:
#             return []

#         try:
#             # Batch encode all queries
#             query_embeddings = self._batch_encode_queries(queries)

#             results = []
#             for i, query_embedding in enumerate(query_embeddings):
#                 query_results = self._search_full(
#                     query_embedding.reshape(1, -1), top_k, 1000
#                 )
#                 results.append(query_results)

#             return results

#         except Exception as e:
#             logger.error(f"Error in batch search: {e}")
#             return [[] for _ in queries]

#     def get_memory_usage(self):
#         """Get current memory usage statistics"""
#         process = psutil.Process()
#         memory_info = process.memory_info()

#         stats = {
#             "cpu_memory_mb": memory_info.rss / 1024 / 1024,
#             "embeddings_shape": (
#                 self.embeddings.shape if self.embeddings is not None else None
#             ),
#             "gpu_available": self.use_gpu,
#         }

#         if self.use_gpu and torch is not None:
#             stats.update(
#                 {
#                     "gpu_memory_allocated_mb": torch.cuda.memory_allocated()
#                     / 1024
#                     / 1024,
#                     "gpu_memory_reserved_mb": torch.cuda.memory_reserved()
#                     / 1024
#                     / 1024,
#                     "gpu_embeddings_loaded": self.embeddings_gpu is not None,
#                 }
#             )

#         return stats

#     def cleanup_gpu_memory(self):
#         """Clean up GPU memory"""
#         if self.use_gpu:
#             if self.embeddings_gpu is not None:
#                 del self.embeddings_gpu
#                 self.embeddings_gpu = None

#             if cp is not None:
#                 cp.get_default_memory_pool().free_all_blocks()

#             if torch is not None:
#                 torch.cuda.empty_cache()

#             gc.collect()
#             logger.info("GPU memory cleaned up")

#     def __del__(self):
#         """Cleanup when object is destroyed"""
#         self.cleanup_gpu_memory()

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import re
from pathlib import Path
import sys
import os
import logging
import psutil
from typing import List, Tuple, Optional, Union
import gc

# GPU acceleration imports
try:
    import cupy as cp
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if GPU_AVAILABLE else "cpu")
except ImportError:
    cp = None
    torch = None
    GPU_AVAILABLE = False
    DEVICE = "cpu"

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

from config.settings import DATASETS, BERT_CONFIG
from config.logging_config import setup_logging
from utils.path_utils import get_embedding_paths

logger = setup_logging("embedding_model")


class EmbeddingModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.model = None
        self.embeddings = None
        self.doc_mapping = None
        self.reverse_mapping = None
        self.loaded = False

        # GPU optimization settings
        self.use_gpu = GPU_AVAILABLE
        self.device = DEVICE
        self.embeddings_gpu = None
        self.batch_size = 64 if self.use_gpu else 32

        logger.info(f"GPU Available: {GPU_AVAILABLE}, Device: {self.device}")

    def load_model(self):
        try:
            # Use dynamic paths
            paths = get_embedding_paths(self.dataset_name)

            # Load BERT model with GPU support
            model_name = BERT_CONFIG["model_name"]
            self.model = SentenceTransformer(model_name, device=str(self.device))
            logger.info(f"BERT model loaded on device: {self.device}")

            # Load components
            self.embeddings = np.load(paths["embeddings"])
            self.doc_mapping = joblib.load(paths["mapping"])

            # Load embeddings to GPU if available
            if self.use_gpu and cp is not None:
                logger.info("Loading embeddings to GPU...")
                self.embeddings_gpu = cp.asarray(self.embeddings)
                logger.info(f"Embeddings loaded to GPU: {self.embeddings_gpu.shape}")

            # Load info if available
            try:
                self.info = joblib.load(paths["info"])
            except FileNotFoundError:
                self.info = {}

            # Create reverse mapping
            if self.dataset_name == "wikir" and isinstance(self.doc_mapping, list):
                # For wikir: doc_mapping is a list, index is idx, value is doc_id
                self.reverse_mapping = {
                    idx: doc_id for idx, doc_id in enumerate(self.doc_mapping)
                }
            else:
                # For other datasets: doc_mapping is a dict {doc_id: idx}
                self.reverse_mapping = {
                    idx: doc_id for doc_id, idx in self.doc_mapping.items()
                }

            self.loaded = True
            logger.info(
                f"BERT model loaded for {self.dataset_name} with GPU acceleration: {self.use_gpu}"
            )

        except Exception as e:
            logger.error(f"Error loading BERT model for {self.dataset_name}: {e}")
            raise

    def preprocess_query(self, query_text):
        if not isinstance(query_text, str):
            return ""

        text = str(query_text).strip()

        if self.dataset_name == "clinical":
            text = re.sub(r"nct\\d+", "", text)
            text = re.sub(r"\\b\\d{4}-\\d{2}-\\d{2}\\b", "", text)
            text = re.sub(
                r"\\b\\d+\\s*(mg|ml|kg|cm|mm|years?|months?|days?|hours?)\\b",
                "DOSAGE",
                text,
            )

        text = re.sub(r"http\\S+|www\\S+", "", text)
        text = re.sub(r"\\S+@\\S+", "", text)
        text = re.sub(r"\\s+", " ", text)

        return text.strip()

    def _gpu_cosine_similarity(self, query_embedding, doc_embeddings, top_k=None):
        """GPU-accelerated cosine similarity computation"""
        if cp is None or not self.use_gpu:
            # Fallback to CPU
            from sklearn.metrics.pairwise import cosine_similarity

            return cosine_similarity(query_embedding, doc_embeddings).flatten()

        # Convert to CuPy arrays
        query_gpu = cp.asarray(query_embedding)
        docs_gpu = (
            cp.asarray(doc_embeddings)
            if not isinstance(doc_embeddings, cp.ndarray)
            else doc_embeddings
        )

        # Normalize vectors
        query_norm = query_gpu / cp.linalg.norm(query_gpu, axis=1, keepdims=True)
        docs_norm = docs_gpu / cp.linalg.norm(docs_gpu, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = cp.dot(query_norm, docs_norm.T).flatten()

        # If top_k is specified, get top results efficiently
        if top_k is not None and top_k < len(similarities):
            top_indices = cp.argpartition(similarities, -top_k)[-top_k:]
            top_similarities = similarities[top_indices]
            # Sort the top results
            sorted_indices = cp.argsort(top_similarities)[::-1]
            return (
                top_similarities[sorted_indices].get(),
                top_indices[sorted_indices].get(),
            )

        return similarities.get()

    def _batch_encode_queries(self, queries: List[str]) -> np.ndarray:
        """Batch encode multiple queries for better GPU utilization"""
        if not isinstance(queries, list):
            queries = [queries]

        processed_queries = [self.preprocess_query(q) for q in queries]

        # Use larger batch size for GPU
        batch_size = self.batch_size if self.use_gpu else 32

        return self.model.encode(
            processed_queries, batch_size=batch_size, show_progress_bar=False
        )

    def search(
        self, query_text, top_k=100, candidate_doc_ids=None, top_n_candidates=1000
    ):
        if not self.loaded:
            self.load_model()

        if not query_text or not query_text.strip():
            return []

        try:
            # Check if model is loaded
            if (
                self.model is None
                or self.embeddings is None
                or self.reverse_mapping is None
            ):
                return []

            # Preprocess and encode query
            query_embedding = self._batch_encode_queries([query_text])

            if candidate_doc_ids is not None:
                # Search within candidate documents
                return self._search_candidates(
                    query_embedding, candidate_doc_ids, top_k
                )
            else:
                # Full search with GPU acceleration
                return self._search_full(query_embedding, top_k, top_n_candidates)

        except Exception as e:
            logger.error(f"Error in BERT search: {e}")
            return []

    def _search_candidates(self, query_embedding, candidate_doc_ids, top_k):
        """Search within specific candidate documents"""
        if self.doc_mapping is not None:
            if isinstance(self.doc_mapping, list):
                # For wikir: doc_id is index in list
                indices = [
                    doc_id
                    for doc_id in candidate_doc_ids
                    if isinstance(doc_id, int) and 0 <= doc_id < len(self.doc_mapping)
                ]
            else:
                indices = [
                    self.doc_mapping[doc_id]
                    for doc_id in candidate_doc_ids
                    if doc_id in self.doc_mapping
                ]
        else:
            indices = []

        if not indices:
            return []

        # Get candidate embeddings
        if self.use_gpu and self.embeddings_gpu is not None:
            candidate_embeddings = self.embeddings_gpu[indices]
            similarities = self._gpu_cosine_similarity(
                query_embedding, candidate_embeddings
            )
        else:
            candidate_embeddings = self.embeddings[indices]
            from sklearn.metrics.pairwise import cosine_similarity

            similarities = cosine_similarity(
                query_embedding, candidate_embeddings
            ).flatten()

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 1e-10:
                doc_id = candidate_doc_ids[idx]
                if doc_id:
                    results.append((doc_id, float(similarities[idx])))

        return results

    def _search_full(self, query_embedding, top_k, top_n_candidates):
        """Full search across all embeddings"""
        if self.use_gpu and self.embeddings_gpu is not None:
            # GPU-accelerated search
            result = self._gpu_cosine_similarity(
                query_embedding, self.embeddings_gpu, top_n_candidates
            )
            if isinstance(result, tuple):
                similarities, top_indices = result
            else:
                similarities = result
                top_indices = np.argsort(similarities)[::-1][:top_n_candidates]
                similarities = similarities[top_indices]
        else:
            # CPU fallback
            from sklearn.metrics.pairwise import cosine_similarity

            similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_n_candidates]
            similarities = similarities[top_indices]
        # Get document IDs for top results
        doc_ids = [self.reverse_mapping.get(idx) for idx in top_indices]
        # Filter and format results
        results = []
        final_indices = np.argsort(similarities)[::-1][:top_k]
        # Debug: Print top similarities
        print(f"[DEBUG] Top similarities: {similarities[final_indices]}")
        for idx in final_indices:
            if similarities[idx] > 1e-10:
                doc_id = doc_ids[idx]
                if doc_id:
                    results.append((doc_id, float(similarities[idx])))
        return results

    def batch_search(
        self, queries: List[str], top_k=100
    ) -> List[List[Tuple[str, float]]]:
        """Batch search for multiple queries - GPU optimized"""
        if not self.loaded:
            self.load_model()

        if not queries:
            return []

        try:
            # Batch encode all queries
            query_embeddings = self._batch_encode_queries(queries)

            results = []
            for i, query_embedding in enumerate(query_embeddings):
                query_results = self._search_full(
                    query_embedding.reshape(1, -1), top_k, 1000
                )
                results.append(query_results)

            return results

        except Exception as e:
            logger.error(f"Error in batch search: {e}")
            return [[] for _ in queries]

    def get_memory_usage(self):
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()

        stats = {
            "cpu_memory_mb": memory_info.rss / 1024 / 1024,
            "embeddings_shape": (
                self.embeddings.shape if self.embeddings is not None else None
            ),
            "gpu_available": self.use_gpu,
        }

        if self.use_gpu and torch is not None:
            stats.update(
                {
                    "gpu_memory_allocated_mb": torch.cuda.memory_allocated()
                    / 1024
                    / 1024,
                    "gpu_memory_reserved_mb": torch.cuda.memory_reserved()
                    / 1024
                    / 1024,
                    "gpu_embeddings_loaded": self.embeddings_gpu is not None,
                }
            )

        return stats

    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        if self.use_gpu:
            if self.embeddings_gpu is not None:
                del self.embeddings_gpu
                self.embeddings_gpu = None

            if cp is not None:
                cp.get_default_memory_pool().free_all_blocks()

            if torch is not None:
                torch.cuda.empty_cache()

            gc.collect()
            logger.info("GPU memory cleaned up")

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup_gpu_memory()
