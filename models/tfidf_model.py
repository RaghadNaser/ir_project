import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys
import os
import re
import gc
import psutil
from typing import List, Tuple, Optional, Union

# GPU acceleration imports
try:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = None
    cp_sparse = None
    GPU_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from config.settings import DATASETS
from config.logging_config import setup_logging
from .inverted_index import InvertedIndex
from utils.path_utils import get_tfidf_paths

logger = setup_logging("tfidf_model")

def smart_preprocessor(text):
    """Custom preprocessor function used during TF-IDF training and search (unified with service)"""
    if not isinstance(text, str):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    return text.lower().strip()

def smart_tokenizer(text):
    """Custom tokenizer function used during TF-IDF training and search (unified with service)"""
    if not isinstance(text, str):
        return []
    text = smart_preprocessor(text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2 and not t.isdigit()]
    return tokens

class TFIDFModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_mapping = None
        self.inverted_index = None
        self.reverse_mapping = None
        self.loaded = False
        
        # GPU optimization settings
        self.use_gpu = GPU_AVAILABLE
        self.tfidf_matrix_gpu = None
        self.batch_size = 512 if self.use_gpu else 128
        
        logger.info(f"TF-IDF GPU acceleration available: {GPU_AVAILABLE}")
    
    def load_model(self):
        try:
            import sys
            main_module = sys.modules['__main__']
            setattr(main_module, 'smart_preprocessor', smart_preprocessor)
            setattr(main_module, 'smart_tokenizer', smart_tokenizer)
            
            # Use dynamic paths
            paths = get_tfidf_paths(self.dataset_name)
            
            # Load vectorizer
            self.vectorizer = joblib.load(paths['vectorizer'])
            
            # Load matrix (joblib or npz)
            if str(paths['matrix']).endswith('.npz'):
                from scipy import sparse
                self.tfidf_matrix = sparse.load_npz(paths['matrix'])
            else:
                self.tfidf_matrix = joblib.load(paths['matrix'])
            
            # Load matrix to GPU if available
            if self.use_gpu and cp is not None:
                logger.info("Loading TF-IDF matrix to GPU...")
                if hasattr(self.tfidf_matrix, 'toarray'):
                    # Convert sparse matrix to GPU
                    self.tfidf_matrix_gpu = cp_sparse.csr_matrix(self.tfidf_matrix)
                else:
                    # Convert dense matrix to GPU
                    self.tfidf_matrix_gpu = cp.asarray(self.tfidf_matrix)
                logger.info(f"TF-IDF matrix loaded to GPU: {self.tfidf_matrix_gpu.shape}")
            
            # Load mapping (joblib or tsv)
            if str(paths['mapping']).endswith('.tsv'):
                import pandas as pd
                mapping_df = pd.read_csv(paths['mapping'], sep='\t')
                self.doc_mapping = dict(zip(mapping_df['doc_id'], mapping_df['idx']))
            else:
                self.doc_mapping = joblib.load(paths['mapping'])
            
            # Create reverse mapping for efficient lookup
            self.reverse_mapping = {idx: doc_id for doc_id, idx in self.doc_mapping.items()}
            
            # Load inverted index
            try:
                from utils.path_utils import get_inverted_index_paths
                index_paths = get_inverted_index_paths(self.dataset_name)
                self.inverted_index = InvertedIndex(self.dataset_name)
                self.inverted_index.build_index()
            except Exception as e:
                logger.warning(f"Could not load inverted index: {e}")
                self.inverted_index = None
            
            self.loaded = True
            logger.info(f"TF-IDF model loaded for {self.dataset_name} with GPU acceleration: {self.use_gpu}")
            
        except Exception as e:
            logger.error(f"Error loading TF-IDF model for {self.dataset_name}: {e}")
            raise
    
    def _gpu_cosine_similarity(self, query_vector, doc_matrix, top_k=None):
        """GPU-accelerated cosine similarity for sparse matrices"""
        if cp is None or not self.use_gpu:
            # Fallback to CPU
            return cosine_similarity(query_vector, doc_matrix).flatten()
        
        # Convert query to GPU
        if hasattr(query_vector, 'toarray'):
            query_gpu = cp_sparse.csr_matrix(query_vector)
        else:
            query_gpu = cp.asarray(query_vector)
        
        # Compute dot product
        if hasattr(doc_matrix, 'toarray') and hasattr(query_gpu, 'toarray'):
            # Both sparse
            dot_products = query_gpu.dot(doc_matrix.T).toarray().flatten()
        else:
            # Mixed or dense
            if hasattr(query_gpu, 'toarray'):
                query_gpu = query_gpu.toarray()
            if hasattr(doc_matrix, 'toarray'):
                doc_matrix = doc_matrix.toarray()
            dot_products = cp.dot(query_gpu, doc_matrix.T).flatten()
        
        # Compute norms
        if hasattr(query_vector, 'toarray'):
            query_norm = cp.sqrt(cp.sum(query_gpu.power(2)))
        else:
            query_norm = cp.linalg.norm(query_gpu)
        
        if hasattr(doc_matrix, 'toarray'):
            doc_norms = cp.sqrt(cp.array([cp.sum(doc_matrix[i].power(2)) for i in range(doc_matrix.shape[0])]))
        else:
            doc_norms = cp.linalg.norm(doc_matrix, axis=1)
        
        # Compute cosine similarity
        similarities = dot_products / (query_norm * doc_norms + 1e-8)
        
        # Return top-k if specified
        if top_k is not None and top_k < len(similarities):
            top_indices = cp.argpartition(similarities, -top_k)[-top_k:]
            top_similarities = similarities[top_indices]
            sorted_indices = cp.argsort(top_similarities)[::-1]
            return top_similarities[sorted_indices].get(), top_indices[sorted_indices].get()
        
        return similarities.get()
    
    def _batch_vectorize_queries(self, queries: List[str]):
        """Batch vectorize multiple queries"""
        if not isinstance(queries, list):
            queries = [queries]
        
        return self.vectorizer.transform(queries)
    
    def search(self, query_text, top_k=100, candidate_doc_ids=None, search_method="hybrid", top_n_candidates=1000, use_dtm_matching=False, return_dict=False):
        if not self.loaded:
            self.load_model()
        if not query_text or not query_text.strip():
            return []
        try:
            # Unified preprocessing and tokenization
            cleaned_query = smart_preprocessor(query_text)
            query_terms = smart_tokenizer(cleaned_query)
            if candidate_doc_ids is not None:
                return self._search_candidates(cleaned_query, candidate_doc_ids, top_k)
            # DTM matching logic (like the service)
            if use_dtm_matching and self.vectorizer is not None and self.reverse_mapping is not None:
                dtm_result = self.dtm_matched_docs(query_terms)
                if dtm_result[0] is not None:
                    submatrix, index_mapping, matched_terms = dtm_result
                    if hasattr(submatrix, 'nnz') and submatrix.nnz > 0:
                        query_vector = self.vectorizer.transform([cleaned_query])
                        cosine_similarities = cosine_similarity(query_vector, submatrix)
                        top_k_indices = np.argsort(-cosine_similarities[0])[:top_k]
                        top_k_doc_indices = [index_mapping[index] for index in top_k_indices]
                        results = []
                        for i, doc_index in enumerate(top_k_doc_indices):
                            similarity_score = cosine_similarities[0][top_k_indices[i]]
                            doc_id = self.reverse_mapping.get(doc_index) if self.reverse_mapping else None
                            if doc_id and similarity_score > 1e-10:
                                if return_dict:
                                    results.append({"doc_id": doc_id, "score": float(similarity_score)})
                                else:
                                    results.append((doc_id, float(similarity_score)))
                        return results
            # Fallback to normal search logic
            if search_method == "inverted_index_only" and self.inverted_index:
                return self._search_with_inverted_index_only(query_terms, top_k)
            elif search_method == "tfidf_only":
                return self._search_with_tfidf_only(cleaned_query, top_k)
            elif search_method == "hybrid" and self.inverted_index:
                return self._search_hybrid(cleaned_query, query_terms, top_k, top_n_candidates=top_n_candidates)
            else:
                return self._search_with_tfidf_only(cleaned_query, top_k)
        except Exception as e:
            logger.error(f"Error in TF-IDF search: {e}")
            return []
    
    def _search_candidates(self, query_text, candidate_doc_ids, top_k):
        """Search within specific candidate documents with GPU acceleration"""
        indices = [self.doc_mapping[doc_id] for doc_id in candidate_doc_ids if doc_id in self.doc_mapping]
        if not indices:
            return []
        
        query_vector = self.vectorizer.transform([query_text])
        
        if self.use_gpu and self.tfidf_matrix_gpu is not None:
            # GPU-accelerated candidate search
            candidate_matrix = self.tfidf_matrix_gpu[indices]
            similarities = self._gpu_cosine_similarity(query_vector, candidate_matrix)
        else:
            # CPU fallback
            candidate_matrix = self.tfidf_matrix[indices]
            similarities = cosine_similarity(query_vector, candidate_matrix).flatten()
        
        doc_ids = [candidate_doc_ids[i] for i in range(len(indices))]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 1e-10:
                doc_id = doc_ids[idx]
                if doc_id:
                    results.append((doc_id, float(similarities[idx])))
        return results
    
    def _search_with_inverted_index_only(self, query_terms, top_k):
        """Search using inverted index only"""
        if not self.inverted_index:
            return []
        
        # Use search with ranking from inverted index
        results = self.inverted_index.search_with_ranking(
            query_terms, 
            top_k=top_k, 
            ranking_method="tfidf"
        )
        
        return results
    
    def _search_with_tfidf_only(self, query_text, top_k):
        """Search using TF-IDF only with GPU acceleration"""
        if self.vectorizer is None:
            return []
        
        query_vector = self.vectorizer.transform([query_text])
        
        if query_vector.nnz == 0:
            return []
        
        # GPU-accelerated full similarity calculation
        if self.use_gpu and self.tfidf_matrix_gpu is not None:
            similarities = self._gpu_cosine_similarity(query_vector, self.tfidf_matrix_gpu)
        else:
            # CPU fallback
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 1e-10:
                doc_id = self.reverse_mapping.get(idx)
                if doc_id:
                    results.append((doc_id, float(similarities[idx])))
        
        return results
    
    def _search_hybrid(self, query_text, query_terms, top_k, top_n_candidates=1000):
        """Hybrid search using both methods with GPU acceleration"""
        if not self.inverted_index or self.vectorizer is None or self.tfidf_matrix is None:
            return self._search_with_tfidf_only(query_text, top_k)
        
        # Get candidates from inverted index
        candidate_docs = self.inverted_index.get_candidate_documents(query_terms, method="union")
        if not candidate_docs:
            return []
        
        # Reduce number of candidates
        candidate_docs = list(candidate_docs)[:top_n_candidates]
        
        # Convert candidate documents to indices in TF-IDF matrix
        candidate_indices = []
        for doc_id in candidate_docs:
            if doc_id in self.doc_mapping:
                candidate_indices.append(self.doc_mapping[doc_id])
        
        if not candidate_indices:
            return []
        
        # Calculate similarity only for candidates with GPU acceleration
        query_vector = self.vectorizer.transform([query_text])
        
        if self.use_gpu and self.tfidf_matrix_gpu is not None:
            candidate_matrix = self.tfidf_matrix_gpu[candidate_indices]
            similarities = self._gpu_cosine_similarity(query_vector, candidate_matrix)
        else:
            candidate_matrix = self.tfidf_matrix[candidate_indices]
            similarities = cosine_similarity(query_vector, candidate_matrix).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 1e-10:
                original_idx = candidate_indices[idx]
                doc_id = self.reverse_mapping.get(original_idx)
                if doc_id:
                    results.append((doc_id, float(similarities[idx])))
        
        logger.debug(f"Hybrid search: {len(candidate_docs)} candidates -> {len(results)} results")
        
        return results
    
    def batch_search(self, queries: List[str], top_k=100) -> List[List[Tuple[str, float]]]:
        """Batch search for multiple queries - GPU optimized"""
        if not self.loaded:
            self.load_model()
        
        if not queries:
            return []
        
        try:
            # Batch vectorize all queries
            query_vectors = self._batch_vectorize_queries(queries)
            
            results = []
            
            # Process in batches for memory efficiency
            batch_size = min(self.batch_size, len(queries))
            
            for i in range(0, len(queries), batch_size):
                batch_vectors = query_vectors[i:i+batch_size]
                batch_results = []
                
                for j in range(batch_vectors.shape[0]):
                    query_vector = batch_vectors[j:j+1]
                    
                    if self.use_gpu and self.tfidf_matrix_gpu is not None:
                        similarities = self._gpu_cosine_similarity(query_vector, self.tfidf_matrix_gpu)
                    else:
                        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                    
                    top_indices = np.argsort(similarities)[::-1][:top_k]
                    
                    query_results = []
                    for idx in top_indices:
                        if similarities[idx] > 1e-10:
                            doc_id = self.reverse_mapping.get(idx)
                            if doc_id:
                                query_results.append((doc_id, float(similarities[idx])))
                    
                    batch_results.append(query_results)
                
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch search: {e}")
            return [[] for _ in queries]
    
    def get_document_info(self, doc_id):
        """Get detailed information about a specific document"""
        if not self.loaded:
            return None
        
        info = {
            'doc_id': doc_id,
            'tfidf_score': 0.0,
            'term_frequencies': {},
            'term_positions': {},
            'term_tfidf_scores': {}
        }
        
        # Get information from inverted index
        if (self.inverted_index is not None and 
            hasattr(self.inverted_index, 'doc_terms') and 
            self.inverted_index.doc_terms is not None):
            
            doc_terms = self.inverted_index.doc_terms
            if isinstance(doc_terms, dict) and doc_id in doc_terms:
                # Get information from inverted index
                for term in doc_terms[doc_id]:
                    info['term_frequencies'][term] = self.inverted_index.get_term_frequency(doc_id, term)
                    info['term_positions'][term] = self.inverted_index.get_term_positions(doc_id, term)
                    info['term_tfidf_scores'][term] = self.inverted_index.get_term_tfidf(doc_id, term)
                    info['tfidf_score'] += info['term_tfidf_scores'][term]
        
        return info
    
    def get_memory_usage(self):
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = {
            'cpu_memory_mb': memory_info.rss / 1024 / 1024,
            'tfidf_matrix_shape': self.tfidf_matrix.shape if self.tfidf_matrix is not None else None,
            'gpu_available': self.use_gpu,
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.vectorizer else 0
        }
        
        if self.use_gpu and cp is not None:
            try:
                stats.update({
                    'gpu_memory_used_mb': cp.get_default_memory_pool().used_bytes() / 1024 / 1024,
                    'gpu_memory_total_mb': cp.get_default_memory_pool().total_bytes() / 1024 / 1024,
                    'gpu_matrix_loaded': self.tfidf_matrix_gpu is not None
                })
            except:
                pass
        
        return stats
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        if self.use_gpu:
            if self.tfidf_matrix_gpu is not None:
                del self.tfidf_matrix_gpu
                self.tfidf_matrix_gpu = None
            
            if cp is not None:
                cp.get_default_memory_pool().free_all_blocks()
            
            gc.collect()
            logger.info("TF-IDF GPU memory cleaned up")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup_gpu_memory()
    
    def dtm_matched_docs(self, query_terms: list) -> tuple:
        """
        DTM matching approach - finds documents containing query terms (unified with service)
        Returns: submatrix, index_mapping, matched_terms_count
        """
        if self.tfidf_matrix is None or self.vectorizer is None:
            return None, {}, 0
        terms_mapping = self.vectorizer.vocabulary_ if self.vectorizer is not None else {}
        term_indices = []
        matched_terms = []
        for term in query_terms:
            if term in terms_mapping:
                term_indices.append(terms_mapping[term])
                matched_terms.append(term)
        if not term_indices:
            return None, {}, 0
        matching_docs = set()
        for term_index in term_indices:
            docs_with_term = self.tfidf_matrix[:, term_index].nonzero()[0]
            matching_docs.update(docs_with_term)
        matching_docs = list(matching_docs)
        if not matching_docs:
            return None, {}, 0
        submatrix = self.tfidf_matrix[matching_docs, :]
        index_mapping = {i: matching_docs[i] for i in range(submatrix.shape[0])}
        return submatrix, index_mapping, len(matched_terms) 