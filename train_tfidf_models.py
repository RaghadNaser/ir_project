#!/usr/bin/env python3
"""
Train TF-IDF models from database data
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import time
from pathlib import Path
import re

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

def load_argsme_data():
    """Load ARGSME data from database"""
    print("📊 Loading ARGSME data...")
    
    conn = sqlite3.connect("data/ir_database_combined.db")
    
    # Load documents
    query = """
    SELECT doc_id, conclusion, premises_texts, source_title, topic, acquisition
    FROM argsme_raw
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"   Loaded {len(df)} ARGSME documents")
    
    # Combine text content
    def combine_text(row):
        parts = []
        if pd.notna(row['conclusion']):
            parts.append(str(row['conclusion']))
        if pd.notna(row['premises_texts']):
            parts.append(str(row['premises_texts']))
        return " ".join(parts)
    
    df['combined_text'] = df.apply(combine_text, axis=1)
    
    # Remove empty documents
    df = df[df['combined_text'].str.strip() != ""]
    print(f"   {len(df)} documents with non-empty text")
    
    return df

def load_wikir_data():
    """Load WIKIR data from database"""
    print("📊 Loading WIKIR data...")
    
    conn = sqlite3.connect("data/ir_database_combined.db")
    
    # Load documents
    query = """
    SELECT doc_id, title, text
    FROM wikir_docs
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"   Loaded {len(df)} WIKIR documents")
    
    # Use text field directly
    df['combined_text'] = df['text'].fillna("")
    
    # Remove empty documents
    df = df[df['combined_text'].str.strip() != ""]
    print(f"   {len(df)} documents with non-empty text")
    
    return df

def train_tfidf_model(dataset_name, df):
    """Train TF-IDF model for a dataset"""
    print(f"\n🔧 Training TF-IDF model for {dataset_name}...")
    
    # Create output directory
    output_dir = Path(f"data/vectors/{dataset_name}/tfidf")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare text data
    texts = df['combined_text'].tolist()
    doc_ids = df['doc_id'].tolist()
    
    print(f"   Training on {len(texts)} documents")
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=50000,  # Limit vocabulary size
        min_df=2,           # Minimum document frequency
        max_df=0.95,        # Maximum document frequency
        ngram_range=(1, 2), # Unigrams and bigrams
        stop_words='english',
        tokenizer=smart_tokenizer,
        preprocessor=smart_preprocessor,
        lowercase=True,
        analyzer='word'
    )
    
    # Fit the vectorizer
    print("   Fitting vectorizer...")
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    print(f"   Matrix shape: {tfidf_matrix.shape}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Create document mapping
    doc_mapping = {i: doc_id for i, doc_id in enumerate(doc_ids)}
    
    # Save models
    print("   Saving models...")
    
    # Save vectorizer
    vectorizer_path = output_dir / f"{dataset_name}_tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    
    # Save matrix
    matrix_path = output_dir / f"{dataset_name}_tfidf_matrix.joblib"
    joblib.dump(tfidf_matrix, matrix_path)
    
    # Save document mapping
    mapping_path = output_dir / f"{dataset_name}_doc_mapping.joblib"
    joblib.dump(doc_mapping, mapping_path)
    
    print(f"   ✅ Models saved to {output_dir}")
    
    return {
        'vectorizer': vectorizer,
        'matrix': tfidf_matrix,
        'doc_mapping': doc_mapping,
        'vocabulary': vectorizer.vocabulary_
    }

def test_model(dataset_name, model_data, test_queries):
    """Test the trained model"""
    print(f"\n🧪 Testing {dataset_name} model...")
    
    vectorizer = model_data['vectorizer']
    matrix = model_data['matrix']
    doc_mapping = model_data['doc_mapping']
    
    for query in test_queries:
        print(f"   Testing query: '{query}'")
        
        try:
            # Transform query
            query_vector = vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, matrix).flatten()
            
            # Get top 3 results
            top_indices = np.argsort(similarities)[::-1][:3]
            
            print(f"   Top results:")
            for i, idx in enumerate(top_indices):
                doc_id = doc_mapping[idx]
                score = similarities[idx]
                print(f"     {i+1}. Doc {doc_id}: {score:.4f}")
                
        except Exception as e:
            print(f"   ❌ Error testing query: {e}")

def main():
    """Main function to train all models"""
    print("🚀 Training TF-IDF Models from Database")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "artificial intelligence",
        "machine learning",
        "climate change",
        "vaccination",
        "ابحث عن الذكاء الاصطناعي"
    ]
    
    # Train ARGSME model
    try:
        argsme_df = load_argsme_data()
        if len(argsme_df) > 0:
            argsme_model = train_tfidf_model("argsme", argsme_df)
            test_model("argsme", argsme_model, test_queries)
        else:
            print("❌ No ARGSME data found")
    except Exception as e:
        print(f"❌ Error training ARGSME model: {e}")
    
    # Train WIKIR model
    try:
        wikir_df = load_wikir_data()
        if len(wikir_df) > 0:
            wikir_model = train_tfidf_model("wikir", wikir_df)
            test_model("wikir", wikir_model, test_queries)
        else:
            print("❌ No WIKIR data found")
    except Exception as e:
        print(f"❌ Error training WIKIR model: {e}")
    
    print("\n" + "=" * 50)
    print("✅ TF-IDF model training completed!")
    print("\n📁 Models saved to:")
    print("   data/vectors/argsme/tfidf/")
    print("   data/vectors/wikir/tfidf/")

if __name__ == "__main__":
    main() 