"""
Simplified TF-IDF Service that works directly with database
Avoids version compatibility issues with pre-trained models
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
import sqlite3
import json
import time
from typing import List, Dict, Optional
from functools import lru_cache
import os

app = FastAPI(title="Simplified TF-IDF Service")

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

class SearchResponse(BaseModel):
    results: List[Dict]
    execution_time: float
    search_method: str

# Global variables
vectorizers = {}
documents = {}
doc_ids = {}

def smart_preprocessor(text):
    """Fast preprocessing for already cleaned data"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    return text.lower().strip()

def get_db_connection():
    return sqlite3.connect("data/ir_database_combined.db", check_same_thread=False)

def load_dataset_data(dataset: str):
    """Load documents from database and create TF-IDF vectorizer"""
    if dataset in vectorizers:
        return
    
    print(f"Loading {dataset} documents from database...")
    start_time = time.time()
    
    try:
        conn = get_db_connection()
        
        if dataset == "argsme":
            # Load argsme documents
            query = "SELECT doc_id, text FROM argsme_docs WHERE text IS NOT NULL AND text != ''"
            df = pd.read_sql_query(query, conn)
        elif dataset == "wikir":
            # Load wikir documents
            query = "SELECT doc_id, text FROM wikir_docs WHERE text IS NOT NULL AND text != ''"
            df = pd.read_sql_query(query, conn)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        conn.close()
        
        if df.empty:
            raise ValueError(f"No documents found for dataset: {dataset}")
        
        # Preprocess documents
        df['processed_text'] = df['text'].apply(smart_preprocessor)
        df = df[df['processed_text'].str.len() > 10]  # Remove very short documents
        
        print(f"Loaded {len(df)} documents for {dataset}")
        
        # Store documents and doc_ids
        documents[dataset] = df['processed_text'].tolist()
        doc_ids[dataset] = df['doc_id'].tolist()
        
        # Create and fit TF-IDF vectorizer
        print(f"Creating TF-IDF vectorizer for {dataset}...")
        vectorizer = TfidfVectorizer(
            max_features=50000,  # Limit features to avoid memory issues
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words='english',
            lowercase=True,
            preprocessor=smart_preprocessor
        )
        
        # Fit the vectorizer on the documents
        tfidf_matrix = vectorizer.fit_transform(documents[dataset])
        vectorizers[dataset] = vectorizer
        
        print(f"{dataset} TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"{dataset} vocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"{dataset} loaded in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"Error loading {dataset} data: {e}")
        raise

@app.get("/")
def root():
    return {"message": "Simplified TF-IDF Service", "status": "running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "datasets_loaded": list(vectorizers.keys()),
        "total_documents": sum(len(docs) for docs in documents.values())
    }

@app.post("/search", response_model=SearchResponse)
def search_simple(req: SearchRequest):
    """Simple TF-IDF search using database data"""
    start_time = time.time()
    
    try:
        # Load dataset if not already loaded
        if req.dataset not in vectorizers:
            load_dataset_data(req.dataset)
        
        # Preprocess query
        cleaned_query = smart_preprocessor(req.query)
        if not cleaned_query.strip():
            return SearchResponse(
                results=[], 
                execution_time=time.time() - start_time,
                search_method="empty_query"
            )
        
        # Get vectorizer and documents
        vectorizer = vectorizers[req.dataset]
        docs = documents[req.dataset]
        doc_ids_list = doc_ids[req.dataset]
        
        # Transform query using the fitted vectorizer
        query_vector = vectorizer.transform([cleaned_query])
        
        # Transform all documents (if not already done)
        doc_vectors = vectorizer.transform(docs)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        
        # Get top K results
        if len(similarities) > req.top_k:
            top_indices = np.argpartition(similarities, -req.top_k)[-req.top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity > 1e-10:  # Only include relevant results
                results.append({
                    "doc_id": doc_ids_list[idx],
                    "score": float(similarity)
                })
        
        results = results[:req.top_k]
        
        execution_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            execution_time=execution_time,
            search_method="simple_tfidf"
        )
        
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        
        return SearchResponse(
            results=[], 
            execution_time=time.time() - start_time,
            search_method="error"
        )

@app.get("/stats")
def get_stats():
    return {
        "datasets": {
            dataset: {
                "documents_loaded": len(docs),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "vectorizer_fitted": vectorizer is not None
            }
            for dataset, (docs, vectorizer) in zip(documents.keys(), zip(documents.values(), vectorizers.values()))
        },
        "method": "Simple TF-IDF with database data",
        "description": "Loads documents from database and creates fresh TF-IDF models"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True) 