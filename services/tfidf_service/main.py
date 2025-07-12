# services/tfidf_service/main_dtm_optimized.py
"""
Optimized TF-IDF Service using DTM matching approach (as suggested by user)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import re
import pandas as pd
import sqlite3
import json
import time
from typing import List, Dict, Optional, Set
from functools import lru_cache
import os
import random

app = FastAPI(title="DTM-Optimized TF-IDF Service")

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
    use_dtm_matching: bool = True  # Use DTM matching for speed

class SearchResponse(BaseModel):
    results: List[Dict]
    execution_time: float
    cache_hit: bool
    search_method: str
    candidates_checked: int
    matched_terms: int

# Global variables for loaded models
vectorizers = {}
matrices = {}  # TF-IDF matrices (DTM)
doc_mappings = {}
vocabularies = {}  # Term to index mapping (like user's 'terms')

# Performance stats
stats = {
    'total_queries': 0,
    'cache_hits': 0,
    'avg_execution_time': 0.0,
    'dtm_matching_queries': 0,
    'full_matrix_queries': 0
}

def smart_preprocessor(text):
    """Fast preprocessing for already cleaned data"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    return text.lower().strip()

def smart_tokenizer(text):
    """Simple tokenizer that matches preprocessing"""
    if not isinstance(text, str) or pd.isna(text):
        return []
    
    text = smart_preprocessor(text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2 and not t.isdigit()]
    return tokens

# Thread-safe database connection
db_conn = None

def get_db_connection():
    global db_conn
    if db_conn is None:
        db_conn = sqlite3.connect("data/ir_database_combined.db", check_same_thread=False)
    return db_conn

@lru_cache(maxsize=1000)
def get_cached_results(query: str, dataset: str, top_k: int, method: str) -> Optional[List[Dict]]:
    """Enhanced LRU cached database lookup"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT results_json FROM cached_results WHERE query=? AND dataset=? AND top_k=? AND extra_info=?", 
                   (query, dataset, top_k, method))
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return None

def cache_results(query: str, dataset: str, top_k: int, method: str, results: List[Dict]):
    """Cache results to database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO cached_results (query, dataset, top_k, extra_info, results_json) VALUES (?, ?, ?, ?, ?)",
                   (query, dataset, top_k, method, json.dumps(results)))
        conn.commit()
    except Exception:
        pass

def dtm_matched_docs(query_terms: List[str], dataset: str) -> tuple:
    """
    DTM matching approach - finds documents containing query terms
    Similar to user's dtm_matched_doc function
    """
    if dataset not in matrices or dataset not in vocabularies:
        return None, {}
    
    # Get TF-IDF matrix and vocabulary (terms mapping)
    tfidf_matrix = matrices[dataset]
    terms_mapping = vocabularies[dataset]  # term -> index mapping
    
    print(f" DTM Matching for terms: {query_terms}")
    
    # Get indices of query terms in vocabulary (like user's term_indices)
    term_indices = []
    matched_terms = []
    for term in query_terms:
        if term in terms_mapping:
            term_indices.append(terms_mapping[term])
            matched_terms.append(term)
            print(f"    Term '{term}' found at index {terms_mapping[term]}")
        else:
            print(f"    Term '{term}' not in vocabulary")
    
    if not term_indices:
        print("   ⚠️  No query terms found in vocabulary")
        return None, {}
    
    # Find documents that contain these terms (like user's matching_docs)
    matching_docs = set()
    for term_index in term_indices:
        # Get documents that have non-zero values for this term
        docs_with_term = tfidf_matrix[:, term_index].nonzero()[0]
        matching_docs.update(docs_with_term)
        print(f"   Term index {term_index}: {len(docs_with_term)} documents")
    
    matching_docs = list(matching_docs)
    print(f" Total matching documents: {len(matching_docs)} (from {tfidf_matrix.shape[0]} total)")
    
    # CAP the number of candidates to avoid MemoryError
    MAX_CANDIDATES = 20000
    if len(matching_docs) > MAX_CANDIDATES:
        print(f"⚠️ Too many candidates ({len(matching_docs)}), reducing to {MAX_CANDIDATES}")
        matching_docs = random.sample(matching_docs, MAX_CANDIDATES)
    
    if not matching_docs:
        return None, {}
    
    # Extract submatrix with only matching documents (like user's submatrix)
    submatrix = tfidf_matrix[matching_docs, :]
    
    # Create index mapping (like user's index_mapping)
    index_mapping = {i: matching_docs[i] for i in range(submatrix.shape[0])}
    
    print(f" Submatrix shape: {submatrix.shape} (vs original {tfidf_matrix.shape})")
    
    return submatrix, index_mapping, len(matched_terms)

def load_dataset_models(dataset: str):
    """Load TF-IDF models and create vocabulary mapping"""
    if dataset in vectorizers:
        return
    
    print(f" Loading {dataset} TF-IDF models...")
    start_time = time.time()
    
    base = f'data/vectors/{dataset}/tfidf'
    
    try:
        if dataset == "argsme":
            import sys
            main_module = sys.modules['__main__']
            setattr(main_module, 'smart_preprocessor', smart_preprocessor)
            
            vectorizers[dataset] = joblib.load(f"data/vectors/argsme/tfidf/argsme_tfidf_vectorizer_improved.joblib")
            matrices[dataset] = joblib.load(f"{base}/argsme_tfidf_matrix_improved.joblib")
            doc_mapping_raw = joblib.load(f"{base}/argsme_doc_mapping_improved.joblib")
            
        elif dataset == "wikir":
            vectorizers[dataset] = joblib.load(f"data/vectors/wikir/tfidf/wikir_tfidf_vectorizer.joblib")
            matrices[dataset] = sparse.load_npz(f"{base}/wikir_tfidf_matrix.npz")
            doc_mapping_raw = joblib.load(f"{base}/wikir_doc_mapping.joblib")
            
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Create vocabulary mapping (term -> index) like user's 'terms'
        vectorizer = vectorizers[dataset]
        vocabularies[dataset] = vectorizer.vocabulary_  # This gives term -> index mapping
        
        # Create efficient bidirectional mapping
        if isinstance(doc_mapping_raw, dict) and "index_to_docid" in doc_mapping_raw:
            doc_mappings[dataset] = doc_mapping_raw
        else:
            doc_mappings[dataset] = {
                "index_to_docid": {v: k for k, v in doc_mapping_raw.items()},
                "docid_to_index": doc_mapping_raw
            }
        
        load_time = time.time() - start_time
        print(f" {dataset} models loaded in {load_time:.2f}s")
        print(f" TF-IDF Matrix: {matrices[dataset].shape}")
        print(f" Vocabulary size: {len(vocabularies[dataset])}")
        
        # Validate doc_mapping coverage
        matrix_size = matrices[dataset].shape[0]
        mapping_size = len(doc_mappings[dataset]["index_to_docid"])
        coverage = mapping_size / matrix_size * 100
        print(f"🔗 Doc mapping coverage: {mapping_size}/{matrix_size} ({coverage:.1f}%)")
        
        if coverage < 95:
            print(f"  Warning: Low doc mapping coverage ({coverage:.1f}%), some results may use fallback IDs")
        
    except Exception as e:
        print(f" Error loading {dataset} models: {e}")
        raise

@app.get("/")
def root():
    return {
        "service": "DTM-Optimized TF-IDF Service",
        "status": "running",
        "loaded_datasets": list(vectorizers.keys()),
        "vocabulary_sizes": {dataset: len(vocab) for dataset, vocab in vocabularies.items()},
        "optimization": "DTM matching for candidate filtering",
        "stats": stats
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "loaded_datasets": list(vectorizers.keys()),
        "matrix_shapes": {dataset: matrices[dataset].shape for dataset in matrices.keys()},
        "cache_info": get_cached_results.cache_info()._asdict()
    }

@app.post("/search", response_model=SearchResponse)
def search_dtm_optimized(req: SearchRequest):
    """
    DTM-optimized search using user's approach:
    1. Find documents containing query terms
    2. Extract submatrix of only those documents  
    3. Compute cosine similarity on submatrix only
    """
    start_time = time.time()
    cache_hit = False
    search_method = "unknown"
    candidates_checked = 0
    matched_terms = 0
    
    try:
        stats['total_queries'] += 1
        
        cache_key = f"dtm_{req.use_dtm_matching}"
        
        # Check cache FIRST
        cached_results = get_cached_results(req.query, req.dataset, req.top_k, cache_key)
        if cached_results:
            stats['cache_hits'] += 1
            return SearchResponse(
                results=cached_results,
                execution_time=time.time() - start_time,
                cache_hit=True,
                search_method="cached",
                candidates_checked=len(cached_results),
                matched_terms=0
            )
        
        # Load models on demand
        if req.dataset not in vectorizers:
            load_dataset_models(req.dataset)
        
        # Preprocess query (like user's process_query)
        cleaned_query = smart_preprocessor(req.query)
        if not cleaned_query.strip():
            return SearchResponse(
                results=[], execution_time=time.time() - start_time,
                cache_hit=False, search_method="empty_query", 
                candidates_checked=0, matched_terms=0
            )
        
        # Vectorize query (like user's query_vector)
        query_vector = vectorizers[req.dataset].transform([cleaned_query])
        query_terms = smart_tokenizer(cleaned_query)
        
        print(f"🔍 Query: '{req.query}' -> Terms: {query_terms}")
        
        results = []
        
        # Method 1: DTM Matching (user's approach)
        if req.use_dtm_matching:
            search_method = "dtm_matching"
            stats['dtm_matching_queries'] += 1
            
            # Get submatrix of documents containing query terms
            dtm_result = dtm_matched_docs(query_terms, req.dataset)
            
            if dtm_result[0] is not None:
                submatrix, index_mapping, matched_terms = dtm_result
                candidates_checked = submatrix.shape[0]
                
                # Calculate cosine similarity on submatrix only (user's approach)
                if submatrix.nnz > 0:  # Check if submatrix has non-zero elements
                    cosine_similarities = cosine_similarity(query_vector, submatrix)
                    
                    # Get top K indices (user's top_ten_indices)
                    top_k_indices = np.argsort(-cosine_similarities[0])[:req.top_k]
                    
                    # Map back to original document indices (user's top_ten_doc_ids)
                    top_k_doc_indices = [index_mapping[index] for index in top_k_indices]
                    
                    # Convert to document IDs
                    index_to_docid = doc_mappings[req.dataset]["index_to_docid"]
                    
                    for i, doc_index in enumerate(top_k_doc_indices):
                        similarity_score = cosine_similarities[0][top_k_indices[i]]
                        if similarity_score > 1e-10:
                            # Convert numpy int to regular int for dictionary lookup
                            doc_index_int = int(doc_index)
                            
                            # Safe lookup with fallback
                            if doc_index_int in index_to_docid:
                                doc_id = index_to_docid[doc_index_int]
                                results.append({
                                    "doc_id": doc_id,
                                    "score": float(similarity_score)
                                })
                            else:
                                # Fallback: use matrix index as doc_id
                                print(f"  Warning: doc_index {doc_index_int} not found in mapping, using fallback")
                                results.append({
                                    "doc_id": f"doc_{doc_index_int}",
                                    "score": float(similarity_score)
                                })
                
                print(f" DTM matching found {len(results)} relevant documents")
            else:
                print("  No matching documents found")
        
        # Method 2: Full Matrix Search (fallback)
        if not results:
            search_method = "full_matrix"
            stats['full_matrix_queries'] += 1
            
            similarities = cosine_similarity(query_vector, matrices[req.dataset]).flatten()
            candidates_checked = len(similarities)
            
            # Get top results efficiently
            if len(similarities) > req.top_k:
                top_indices = np.argpartition(similarities, -req.top_k)[-req.top_k:]
                top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            else:
                top_indices = np.argsort(similarities)[::-1]
            
            # Build results
            index_to_docid = doc_mappings[req.dataset]["index_to_docid"]
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > 1e-10:
                    # Convert numpy int to regular int for dictionary lookup
                    idx_int = int(idx)
                    
                    # Safe lookup with fallback
                    if idx_int in index_to_docid:
                        doc_id = index_to_docid[idx_int]
                        results.append({
                            "doc_id": doc_id,
                            "score": float(similarity)
                        })
                    else:
                        # Fallback: use matrix index as doc_id
                        print(f"  Warning: doc_index {idx_int} not found in mapping, using fallback")
                        results.append({
                            "doc_id": f"doc_{idx_int}",
                            "score": float(similarity)
                        })
            
            results = results[:req.top_k]
        
        # Cache results
        if results:
            cache_results(req.query, req.dataset, req.top_k, cache_key, results)
        
        execution_time = time.time() - start_time
        stats['avg_execution_time'] = (
            (stats['avg_execution_time'] * (stats['total_queries'] - 1) + execution_time) / 
            stats['total_queries']
        )
        
        return SearchResponse(
            results=results,
            execution_time=execution_time,
            cache_hit=cache_hit,
            search_method=search_method,
            candidates_checked=candidates_checked,
            matched_terms=matched_terms
        )
        
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide more specific error information
        error_method = "error"
        if "KeyError" in str(e):
            error_method = "mapping_error"
            print(f"💡 Hint: Document mapping issue - using fallback strategy")
        
        return SearchResponse(
            results=[], execution_time=time.time() - start_time,
            cache_hit=False, search_method=error_method, 
            candidates_checked=0, matched_terms=0
        )

@app.get("/stats")
def get_stats():
    return {
        "performance": stats,
        "cache": get_cached_results.cache_info()._asdict(),
        "datasets": {
            dataset: {
                "matrix_shape": matrices[dataset].shape,
                "vocabulary_size": len(vocabularies[dataset])
            }
            for dataset in vectorizers.keys()
        },
        "optimization_info": {
            "method": "DTM matching approach (user suggested)",
            "description": "Extract submatrix of documents containing query terms",
            "advantage": "Much faster than full matrix search"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003) 