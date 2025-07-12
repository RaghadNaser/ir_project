import pandas as pd
import numpy as np
import joblib
from collections import Counter
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import tqdm
import psutil
import os
import sys

# Add the path to access smart_preprocessor
sys.path.append('services/hybrid_service')

def smart_preprocessor(text):
    """Optimized preprocessing for already cleaned ARGSME data"""
    if not isinstance(text, str) or pd.isna(text):
        return ""

    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = re.sub(r'[^\w\s\-]', ' ', text)  # Remove all non-word characters except hyphens
    return text.lower().strip()

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_existing_tfidf(base_path="data/vectors/argsme/tfidf/"):
    """Load existing TF-IDF files"""
    print("[INFO] Loading existing TF-IDF files...")
    
    try:
        # Load TF-IDF matrix
        print("[INFO] Loading TF-IDF matrix...")
        tfidf_matrix = joblib.load(f"{base_path}argsme_tfidf_matrix_improved.joblib")
        print(f"[INFO] Loaded TF-IDF matrix with shape: {tfidf_matrix.shape}")
        
        # Load vectorizer
        print("[INFO] Loading TF-IDF vectorizer...")
        vectorizer = joblib.load(f"{base_path}argsme_tfidf_vectorizer_improved.joblib")
        print(f"[INFO] Loaded TF-IDF vectorizer with {len(vectorizer.get_feature_names_out())} features")
        
        # Load document mapping
        print("[INFO] Loading document mapping...")
        doc_mapping = joblib.load(f"{base_path}argsme_doc_mapping_improved.joblib")
        print(f"[INFO] Loaded document mapping with {len(doc_mapping)} documents")
        
        return tfidf_matrix, vectorizer, doc_mapping
    except Exception as e:
        print(f"[ERROR] Failed to load existing TF-IDF files: {e}")
        print(f"[INFO] Will try to create new TF-IDF from existing preprocessed data...")
        return None, None, None

def create_tfidf_from_scratch(texts, max_features=5000):
    """Create TF-IDF from scratch if loading fails"""
    print("[INFO] Creating TF-IDF from scratch using preprocessed texts...")
    
    # Create new TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        stop_words='english',
        min_df=2,
        max_df=0.8,
        preprocessor=smart_preprocessor,
        sublinear_tf=True
    )
    
    # Fit and transform texts
    print("[INFO] Fitting TF-IDF vectorizer...")
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Create document mapping
    doc_mapping = {i: i for i in range(len(texts))}
    
    print(f"[INFO] Created TF-IDF matrix with shape: {tfidf_matrix.shape}")
    print(f"[INFO] Created vectorizer with {len(vectorizer.get_feature_names_out())} features")
    
    return tfidf_matrix, vectorizer, doc_mapping

def extract_keywords_from_tfidf(tfidf_matrix, vectorizer, n_keywords=5):
    """Extract keywords using existing TF-IDF matrix"""
    print(f"[INFO] Extracting keywords from existing TF-IDF matrix...")
    
    feature_names = vectorizer.get_feature_names_out()
    keywords_per_doc = []
    all_keywords = []
    
    print("[INFO] Processing documents for keyword extraction...")
    for i in tqdm(range(tfidf_matrix.shape[0]), desc="Extracting keywords"):
        # Get TF-IDF scores for this document
        doc_tfidf = tfidf_matrix[i].toarray()[0]
        
        # Get indices of top keywords
        top_indices = doc_tfidf.argsort()[-n_keywords:][::-1]
        
        # Get keywords and their scores
        doc_keywords = [(feature_names[idx], doc_tfidf[idx]) for idx in top_indices if doc_tfidf[idx] > 0]
        
        keywords_per_doc.append(doc_keywords)
        all_keywords.extend([kw for kw, _ in doc_keywords])
    
    return all_keywords, keywords_per_doc

def find_document_topics(tfidf_matrix, vectorizer, top_n=10):
    """Find most representative topics using document-term analysis"""
    print(f"[INFO] Finding document topics using TF-IDF analysis...")
    
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate average TF-IDF scores across all documents
    avg_tfidf = tfidf_matrix.mean(axis=0).A1
    
    # Get top global terms
    top_indices = avg_tfidf.argsort()[-top_n * 10:][::-1]
    global_topics = [(feature_names[idx], avg_tfidf[idx]) for idx in top_indices]
    
    return global_topics

def analyze_topic_coverage(tfidf_matrix, vectorizer, keywords_per_doc):
    """Analyze how topics are distributed across documents"""
    print("[INFO] Analyzing topic coverage across documents...")
    
    feature_names = vectorizer.get_feature_names_out()
    
    # Count how many documents each term appears in
    doc_frequency = {}
    for i in tqdm(range(tfidf_matrix.shape[0]), desc="Analyzing coverage"):
        doc_tfidf = tfidf_matrix[i].toarray()[0]
        for j, score in enumerate(doc_tfidf):
            if score > 0:
                term = feature_names[j]
                if term not in doc_frequency:
                    doc_frequency[term] = 0
                doc_frequency[term] += 1
    
    # Calculate coverage statistics
    total_docs = tfidf_matrix.shape[0]
    coverage_stats = {}
    for term, count in doc_frequency.items():
        coverage_stats[term] = {
            'doc_count': count,
            'coverage_ratio': count / total_docs,
            'is_common': count > total_docs * 0.1,  # Appears in >10% of docs
            'is_rare': count < total_docs * 0.01     # Appears in <1% of docs
        }
    
    return coverage_stats

def main():
    print("=" * 60)
    print("TOPIC EXTRACTION FOR ARGSME (NO CLUSTERING)")
    print("Optimized for argumentative datasets")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load ARGSME documents
    print("[INFO] Loading ARGSME documents...")
    df = pd.read_csv("data/vectors/argsme/processed/ARGSME_cleaned_docs.tsv", sep="\t")
    texts = df["processed_text"].astype(str).tolist()
    print(f"[INFO] Loaded {len(texts)} documents.")
    print(f"[INFO] Memory usage: {get_memory_usage():.1f} MB")
    
    # Load precomputed BERT embeddings (for future use)
    print("[INFO] Loading precomputed BERT embeddings...")
    embeddings = np.load("data/vectors/argsme/embedding/argsme_bert_embeddings.npy")
    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    print(f"[INFO] Memory usage: {get_memory_usage():.1f} MB")
    
    if len(texts) != embeddings.shape[0]:
        print(f"[WARNING] Number of texts ({len(texts)}) != number of embeddings ({embeddings.shape[0]}), trimming texts.")
        texts = texts[:embeddings.shape[0]]
    
    print(f"[INFO] Processing {len(texts)} documents with topic extraction methods...")
    
    # Method 1: Load existing TF-IDF and extract keywords
    print("\n" + "="*50)
    print("METHOD 1: TF-IDF KEYWORD EXTRACTION")
    print("="*50)
    
    method1_start = time.time()
    
    # Load existing TF-IDF files
    tfidf_matrix, vectorizer, doc_mapping = load_existing_tfidf()
    
    if tfidf_matrix is None:
        # Fallback: create TF-IDF from scratch using preprocessed data
        print("[INFO] Creating TF-IDF from scratch as fallback...")
        tfidf_matrix, vectorizer, doc_mapping = create_tfidf_from_scratch(texts, max_features=5000)
    
    if tfidf_matrix is not None:
        # Extract keywords using TF-IDF
        all_keywords, keywords_per_doc = extract_keywords_from_tfidf(
            tfidf_matrix, vectorizer, n_keywords=5
        )
        method1_time = time.time() - method1_start
        
        print(f"[INFO] Method 1 completed in {method1_time:.1f} seconds")
        print(f"[INFO] Extracted {len(all_keywords)} total keywords")
        print(f"[INFO] Memory usage: {get_memory_usage():.1f} MB")
    else:
        print("[ERROR] Could not create TF-IDF. Cannot proceed.")
        return
    
    # Method 2: Global topic analysis
    print("\n" + "="*50)
    print("METHOD 2: GLOBAL TOPIC ANALYSIS")
    print("="*50)
    
    method2_start = time.time()
    global_topics = find_document_topics(tfidf_matrix, vectorizer, top_n=50)
    method2_time = time.time() - method2_start
    
    print(f"[INFO] Method 2 completed in {method2_time:.1f} seconds")
    print(f"[INFO] Found {len(global_topics)} global topics")
    print(f"[INFO] Memory usage: {get_memory_usage():.1f} MB")
    
    # Method 3: Topic coverage analysis
    print("\n" + "="*50)
    print("METHOD 3: TOPIC COVERAGE ANALYSIS")
    print("="*50)
    
    method3_start = time.time()
    coverage_stats = analyze_topic_coverage(tfidf_matrix, vectorizer, keywords_per_doc)
    method3_time = time.time() - method3_start
    
    print(f"[INFO] Method 3 completed in {method3_time:.1f} seconds")
    print(f"[INFO] Analyzed coverage for {len(coverage_stats)} terms")
    print(f"[INFO] Memory usage: {get_memory_usage():.1f} MB")
    
    # Frequency analysis
    print("\n" + "="*50)
    print("FREQUENCY ANALYSIS")
    print("="*50)
    
    print("[INFO] Performing frequency analysis...")
    keyword_counter = Counter(all_keywords)
    top_global_topics = keyword_counter.most_common(200)
    
    # Rank topics by frequency and importance
    ranked_topics = []
    for keyword, count in top_global_topics:
        frequency_ratio = count / len(texts)
        coverage_info = coverage_stats.get(keyword, {})
        ranked_topics.append((keyword, count, frequency_ratio, coverage_info))
    
    ranked_topics.sort(key=lambda x: x[1], reverse=True)
    
    # Advanced topic analysis
    print("[INFO] Performing advanced topic analysis...")
    topic_texts = [topic for topic, _, _, _ in ranked_topics[:50]]
    similar_topics = []
    
    if len(topic_texts) > 1:
        topic_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        topic_matrix = topic_vectorizer.fit_transform(topic_texts)
        topic_similarities = cosine_similarity(topic_matrix)
        
        # Find similar topics
        for i in range(len(topic_similarities)):
            for j in range(i+1, len(topic_similarities)):
                if topic_similarities[i][j] > 0.3:
                    similar_topics.append((topic_texts[i], topic_texts[j], topic_similarities[i][j]))
    
    # Save results
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    total_time = time.time() - start_time
    
    # Filter topics by quality
    high_quality_topics = [t for t in ranked_topics if t[2] > 0.001]  # Appears in >0.1% of docs
    medium_coverage_topics = [t for t in ranked_topics if 0.01 < t[2] < 0.5]  # 1-50% coverage
    
    results = {
        'global_topics': ranked_topics[:50],
        'high_quality_topics': high_quality_topics[:100],
        'medium_coverage_topics': medium_coverage_topics[:50],
        'keyword_frequency': dict(top_global_topics[:300]),
        'keywords_per_doc': keywords_per_doc,
        'global_tfidf_topics': global_topics[:50],
        'coverage_statistics': coverage_stats,
        'total_documents': len(texts),
        'total_documents_original': len(df),
        'total_keywords_extracted': len(all_keywords),
        'unique_keywords': len(keyword_counter),
        'processing_time_seconds': total_time,
        'method1_time': method1_time,
        'method2_time': method2_time,
        'method3_time': method3_time,
        'processing_speed_docs_per_sec': len(texts) / total_time,
        'similar_topics': similar_topics,
        'performance_metrics': {
            'total_time': total_time,
            'keyword_extraction_time': method1_time,
            'global_analysis_time': method2_time,
            'coverage_analysis_time': method3_time,
            'docs_per_second': len(texts) / total_time,
            'peak_memory_mb': get_memory_usage(),
            'used_existing_tfidf': True,
            'clustering_skipped': True,
            'dataset_type': 'argumentative'
        }
    }
    
    # Save with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"services/topic_detection_service/models/topics_argsme_no_clustering_{timestamp}.joblib"
    joblib.dump(results, output_file)
    
    # Also save the main file
    main_output_file = "services/topic_detection_service/models/keybert_argsme_topics_enhanced.joblib"
    joblib.dump(results, main_output_file)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"[INFO] Topic extraction completed without clustering")
    print(f"[INFO] Documents processed: {len(texts)}")
    print(f"[INFO] Total keywords extracted: {len(all_keywords)}")
    print(f"[INFO] Unique keywords: {len(keyword_counter)}")
    print(f"[INFO] High quality topics: {len(high_quality_topics)}")
    print(f"[INFO] Medium coverage topics: {len(medium_coverage_topics)}")
    print(f"[INFO] Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"[INFO] Processing speed: {len(texts)/total_time:.1f} docs/sec")
    print(f"[INFO] Peak memory usage: {get_memory_usage():.1f} MB")
    print(f"[INFO] Results saved to: {output_file}")
    
    if ranked_topics:
        print(f"\n[INFO] Top 15 global topics:")
        for i, (topic, count, ratio, coverage) in enumerate(ranked_topics[:15]):
            coverage_pct = coverage.get('coverage_ratio', 0) * 100
            print(f"  {i+1}. {topic} (count: {count}, freq: {ratio:.4f}, coverage: {coverage_pct:.1f}%)")
    
    if similar_topics:
        print(f"\n[INFO] Found {len(similar_topics)} similar topic pairs:")
        for topic1, topic2, similarity in similar_topics[:5]:
            print(f"  - '{topic1}' ~ '{topic2}' (similarity: {similarity:.3f})")
    
    print(f"\n[INFO] Topic quality distribution:")
    print(f"  - High quality (>0.1% docs): {len(high_quality_topics)} topics")
    print(f"  - Medium coverage (1-50%): {len(medium_coverage_topics)} topics")
    common_topics = len([t for t in ranked_topics if t[2] > 0.1])
    print(f"  - Common topics (>10%): {common_topics} topics")

if __name__ == "__main__":
    main() 