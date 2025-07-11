from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import joblib
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Topic Detection Service",
    description="API for topic detection using trained ARGSME and WikiIR topic models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded models
argsme_model = None
wikir_model = None
argsme_vectorizer = None
wikir_vectorizer = None

class QueryRequest(BaseModel):
    query: str
    dataset: Optional[str] = "argsme"  # "argsme" or "wikir"
    max_topics: Optional[int] = 10
    min_relevance_score: Optional[float] = 0.1

class TopicResponse(BaseModel):
    topic: str
    relevance_score: float
    frequency: int
    coverage_ratio: float
    doc_count: Optional[int] = None

class TopicDetectionResponse(BaseModel):
    query: str
    dataset: str
    detected_topics: List[TopicResponse]
    similar_topics: List[Dict[str, Any]]
    processing_time: float

class TopicSuggestionResponse(BaseModel):
    dataset: str
    suggestions: List[str]
    related_topics: List[Dict[str, Any]]
    topic_categories: List[str]

def smart_preprocessor(text):
    """Optimized preprocessing for already cleaned data"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = re.sub(r'[^\w\s\-]', ' ', text)  # Remove all non-word characters except hyphens
    return text.lower().strip()

def load_topic_models():
    """Load both ARGSME and WikiIR topic models"""
    global argsme_model, wikir_model, argsme_vectorizer, wikir_vectorizer
    
    # Load ARGSME model
    try:
        argsme_paths = [
            Path("models/keybert_argsme_topics_enhanced.joblib"),
            Path("services/topic_detection_service/models/keybert_argsme_topics_enhanced.joblib"),
            Path("../../models/keybert_argsme_topics_enhanced.joblib"),
        ]
        
        argsme_model_path = None
        for path in argsme_paths:
            if path.exists():
                argsme_model_path = path
                break
        
        if argsme_model_path:
            logger.info(f"Loading ARGSME topic model from: {argsme_model_path}")
            argsme_model = joblib.load(argsme_model_path)
            logger.info(f"ARGSME model loaded with {len(argsme_model.get('global_topics', []))} global topics")
        else:
            logger.warning("ARGSME topic model not found")
            
    except Exception as e:
        logger.error(f"Failed to load ARGSME topic model: {e}")
    
    # Load WikiIR model
    try:
        wikir_paths = [
            Path("models/keybert_wikir_topics_enhanced.joblib"),
            Path("services/topic_detection_service/models/keybert_wikir_topics_enhanced.joblib"),
            Path("../../models/keybert_wikir_topics_enhanced.joblib"),
        ]
        
        wikir_model_path = None
        for path in wikir_paths:
            if path.exists():
                wikir_model_path = path
                break
        
        if wikir_model_path:
            logger.info(f"Loading WikiIR topic model from: {wikir_model_path}")
            wikir_model = joblib.load(wikir_model_path)
            logger.info(f"WikiIR model loaded with {len(wikir_model.get('global_topics', []))} global topics")
        else:
            logger.warning("WikiIR topic model not found")
            
    except Exception as e:
        logger.error(f"Failed to load WikiIR topic model: {e}")

def get_model(dataset: str):
    """Get the appropriate model based on dataset"""
    if dataset.lower() == "argsme":
        return argsme_model
    elif dataset.lower() == "wikir":
        return wikir_model
    else:
        raise HTTPException(status_code=400, detail="Invalid dataset. Choose 'argsme' or 'wikir'")

def extract_query_topics(query: str, dataset: str, max_topics: int = 10, min_relevance_score: float = 0.1):
    """Extract topics from a query using the appropriate trained model with enhanced matching"""
    topic_model = get_model(dataset)
    
    if not topic_model:
        raise HTTPException(status_code=500, detail=f"{dataset.upper()} topic model not loaded")
    
    try:
        # Preprocess query
        processed_query = smart_preprocessor(query)
        
        # Get global topics from model
        global_topics = topic_model.get('global_topics', [])
        keyword_frequency = topic_model.get('keyword_frequency', {})
        
        # Enhanced matching: split query into words and also keep full query
        query_words = set(processed_query.split())
        query_full = processed_query.lower()
        detected_topics = []
        
        for topic_data in global_topics:
            if len(topic_data) >= 3:
                topic, count, ratio = topic_data[0], topic_data[1], topic_data[2]
                coverage_info = topic_data[3] if len(topic_data) > 3 else {}
            else:
                topic, count, ratio = topic_data[0], topic_data[1], 0.0
                coverage_info = {}
            
            topic_lower = topic.lower()
            topic_words = set(topic.split())
            relevance_score = 0.0
            
            # Method 1: Exact word overlap (original method)
            word_overlap = len(query_words.intersection(topic_words))
            if word_overlap > 0:
                relevance_score = max(relevance_score, word_overlap / len(topic_words))
            
            # Method 2: Partial word matching (NEW)
            partial_matches = 0
            for query_word in query_words:
                for topic_word in topic_words:
                    # Check if query word is contained in topic word or vice versa
                    if len(query_word) >= 3:  # Only for words with 3+ characters
                        if query_word in topic_word or topic_word in query_word:
                            # Give higher score for longer matches
                            match_ratio = min(len(query_word), len(topic_word)) / max(len(query_word), len(topic_word))
                            partial_matches += match_ratio
            
            if partial_matches > 0:
                partial_score = partial_matches / len(topic_words)
                relevance_score = max(relevance_score, partial_score * 0.8)  # Slightly lower weight than exact match
            
            # Method 3: Substring matching (NEW)
            substring_score = 0.0
            if query_full in topic_lower:
                substring_score = len(query_full) / len(topic_lower)
                relevance_score = max(relevance_score, substring_score)
            elif topic_lower in query_full:
                substring_score = len(topic_lower) / len(query_full)
                relevance_score = max(relevance_score, substring_score * 0.9)
            
            # Method 4: Character-level similarity for short queries (NEW)
            if len(query_full) <= 5:  # For short queries like "sch"
                for topic_word in topic_words:
                    if len(topic_word) >= len(query_full):
                        # Check if query is a prefix of the topic word
                        if topic_word.startswith(query_full):
                            prefix_score = len(query_full) / len(topic_word)
                            relevance_score = max(relevance_score, prefix_score * 0.7)
                        # Check character overlap
                        elif query_full in topic_word:
                            char_score = len(query_full) / len(topic_word)
                            relevance_score = max(relevance_score, char_score * 0.6)
            
            # Boost score for exact matches
            if topic_lower == query_full:
                relevance_score *= 2.0
            elif query_full in topic_lower:
                relevance_score *= 1.5
            
            # Boost score for high frequency topics
            if count > 100:
                relevance_score *= 1.2
            elif count > 50:
                relevance_score *= 1.1
            
            # Apply minimum relevance threshold
            if relevance_score >= min_relevance_score:
                detected_topics.append({
                    'topic': topic,
                    'relevance_score': min(relevance_score, 1.0),
                    'frequency': count,
                    'coverage_ratio': ratio,
                    'doc_count': coverage_info.get('doc_count', 0)
                })
        
        # Sort by relevance score
        detected_topics.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Limit results
        detected_topics = detected_topics[:max_topics]
        
        # Find similar topics - USING ACTUAL TRAINED DATA
        similar_topics = []
        
        # Method 1: Use pre-computed similar topics from training
        if 'similar_topics' in topic_model and topic_model['similar_topics']:
            for topic1, topic2, similarity in topic_model['similar_topics']:
                # Include all pre-computed similarities, not just those matching detected topics
                similar_topics.append({
                    'topic1': topic1,
                    'topic2': topic2,
                    'similarity': float(similarity),
                    'type': 'pre-computed'
                })
        
        # Method 2: Use keyword frequency and coverage data from training
        if detected_topics and 'keyword_frequency' in topic_model and 'coverage_statistics' in topic_model:
            keyword_frequency = topic_model['keyword_frequency']
            coverage_stats = topic_model['coverage_statistics']
            global_topics = topic_model.get('global_topics', [])
            
            # Get all topics from the actual trained model
            all_topics = [topic_data[0] for topic_data in global_topics[:200]]
            detected_topic_names = [dt['topic'] for dt in detected_topics]
            
            # Find semantically related topics using actual training data
            for detected_topic in detected_topic_names:
                detected_words = set(detected_topic.lower().split())
                
                for candidate_topic in all_topics:
                    if candidate_topic not in detected_topic_names:
                        candidate_words = set(candidate_topic.lower().split())
                        
                        # Calculate semantic similarity using actual data
                        semantic_score = 0.0
                        
                        # Method 2a: Word overlap with frequency weighting
                        word_overlap = len(detected_words.intersection(candidate_words))
                        if word_overlap > 0:
                            # Weight by actual frequency from training
                            detected_freq = keyword_frequency.get(detected_topic, 1)
                            candidate_freq = keyword_frequency.get(candidate_topic, 1)
                            freq_weight = min(detected_freq, candidate_freq) / max(detected_freq, candidate_freq)
                            semantic_score = max(semantic_score, (word_overlap / max(len(detected_words), len(candidate_words))) * freq_weight)
                        
                        # Method 2b: Coverage-based similarity using actual coverage stats
                        if detected_topic in coverage_stats and candidate_topic in coverage_stats:
                            det_coverage = coverage_stats[detected_topic].get('coverage_ratio', 0)
                            cand_coverage = coverage_stats[candidate_topic].get('coverage_ratio', 0)
                            
                            # Topics with similar coverage ratios are more likely to be related
                            coverage_diff = abs(det_coverage - cand_coverage)
                            if coverage_diff < 0.01:  # Similar coverage patterns
                                semantic_score = max(semantic_score, 0.6)
                            elif coverage_diff < 0.05:
                                semantic_score = max(semantic_score, 0.4)
                        
                        # Method 2c: Partial word matching with actual frequency data
                        partial_matches = 0
                        for det_word in detected_words:
                            for cand_word in candidate_words:
                                if len(det_word) >= 3 and len(cand_word) >= 3:
                                    if det_word in cand_word or cand_word in det_word:
                                        # Weight partial matches by frequency in actual data
                                        det_word_freq = keyword_frequency.get(det_word, 0)
                                        cand_word_freq = keyword_frequency.get(cand_word, 0)
                                        if det_word_freq > 0 and cand_word_freq > 0:
                                            match_ratio = min(len(det_word), len(cand_word)) / max(len(det_word), len(cand_word))
                                            freq_boost = min(det_word_freq, cand_word_freq) / 100  # Normalize frequency
                                            partial_matches += match_ratio * (1 + freq_boost)
                        
                        if partial_matches > 0:
                            partial_score = partial_matches / max(len(detected_words), len(candidate_words))
                            semantic_score = max(semantic_score, partial_score * 0.8)
                        
                        # Method 2d: Co-occurrence analysis using actual topic data
                        # Check if topics frequently co-occur in the same documents (using coverage stats)
                        if detected_topic in coverage_stats and candidate_topic in coverage_stats:
                            det_doc_count = coverage_stats[detected_topic].get('doc_count', 0)
                            cand_doc_count = coverage_stats[candidate_topic].get('doc_count', 0)
                            total_docs = topic_model.get('total_documents', 1)
                            
                            # Estimate co-occurrence probability
                            det_prob = det_doc_count / total_docs
                            cand_prob = cand_doc_count / total_docs
                            expected_cooccur = det_prob * cand_prob * total_docs
                            
                            # Boost score for topics that might co-occur
                            if expected_cooccur > 10:  # Threshold for meaningful co-occurrence
                                semantic_score = max(semantic_score, 0.3)
                        
                        # Add to similar topics if score is high enough
                        if semantic_score >= 0.2:
                            similar_topics.append({
                                'topic1': detected_topic,
                                'topic2': candidate_topic,
                                'similarity': semantic_score,
                                'type': 'frequency-based'
                            })
        
        # Method 3: Query-based similarity using actual topic frequencies
        if 'global_topics' in topic_model:
            query_similar = []
            global_topics = topic_model['global_topics']
            keyword_frequency = topic_model.get('keyword_frequency', {})
            
            for topic_data in global_topics[:100]:
                topic = topic_data[0]
                topic_freq = topic_data[1] if len(topic_data) > 1 else 0
                
                if topic not in [dt['topic'] for dt in detected_topics]:
                    topic_words = set(topic.lower().split())
                    
                    # Calculate similarity to original query using frequency weighting
                    query_similarity = 0.0
                    
                    for query_word in query_words:
                        for topic_word in topic_words:
                            if len(query_word) >= 3 and len(topic_word) >= 3:
                                if query_word in topic_word or topic_word in query_word:
                                    match_ratio = min(len(query_word), len(topic_word)) / max(len(query_word), len(topic_word))
                                    # Weight by actual topic frequency from training
                                    freq_weight = min(topic_freq / 100, 1.0)  # Normalize frequency
                                    query_similarity = max(query_similarity, match_ratio * (1 + freq_weight))
                    
                    if query_similarity >= 0.3:
                        query_similar.append({
                            'topic1': processed_query,
                            'topic2': topic,
                            'similarity': query_similarity,
                            'type': 'query-frequency'
                        })
            
            similar_topics.extend(query_similar)
        
        # Remove duplicates and sort by similarity
        seen_pairs = set()
        unique_similar = []
        for item in sorted(similar_topics, key=lambda x: x['similarity'], reverse=True):
            pair_key = tuple(sorted([item['topic1'], item['topic2']]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                unique_similar.append(item)
        
        return detected_topics, unique_similar[:15]
        
    except Exception as e:
        logger.error(f"Error extracting topics from query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load both topic models on startup"""
    try:
        load_topic_models()
        logger.info("Topic Detection Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Topic Detection Service",
        "version": "1.0.0",
        "description": "API for topic detection using trained ARGSME and WikiIR topic models",
        "supported_datasets": ["argsme", "wikir"],
        "endpoints": {
            "/detect-topics": "POST - Detect topics in a query",
            "/suggest-topics": "GET - Get topic suggestions",
            "/health": "GET - Health check",
            "/model-info": "GET - Get model information",
            "/datasets": "GET - List available datasets"
        }
    }

@app.post("/detect-topics", response_model=TopicDetectionResponse)
async def detect_topics(request: QueryRequest):
    """Detect topics in a given query"""
    import time
    start_time = time.time()
    
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        dataset = request.dataset.lower() if request.dataset else "argsme"
        if dataset not in ["argsme", "wikir"]:
            raise HTTPException(status_code=400, detail="Invalid dataset. Choose 'argsme' or 'wikir'")
        
        detected_topics, similar_topics = extract_query_topics(
            request.query,
            dataset,
            max_topics=request.max_topics or 10,
            min_relevance_score=request.min_relevance_score or 0.1
        )
        
        processing_time = time.time() - start_time
        
        # Convert to response format
        topic_responses = [
            TopicResponse(
                topic=topic['topic'],
                relevance_score=topic['relevance_score'],
                frequency=topic['frequency'],
                coverage_ratio=topic['coverage_ratio'],
                doc_count=topic.get('doc_count', 0)
            )
            for topic in detected_topics
        ]
        
        return TopicDetectionResponse(
            query=request.query,
            dataset=dataset,
            detected_topics=topic_responses,
            similar_topics=similar_topics,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_topics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/suggest-topics", response_model=TopicSuggestionResponse)
async def suggest_topics(
    dataset: str = Query("argsme", description="Dataset to use: 'argsme' or 'wikir'"),
    category: Optional[str] = None, 
    limit: int = 20
):
    """Get topic suggestions"""
    try:
        topic_model = get_model(dataset)
        if not topic_model:
            raise HTTPException(status_code=500, detail=f"{dataset.upper()} topic model not loaded")
        
        global_topics = topic_model.get('global_topics', [])
        
        # Get top topics as suggestions
        suggestions = []
        related_topics = []
        
        for topic_data in global_topics[:limit]:
            if len(topic_data) >= 3:
                topic, count, ratio = topic_data[0], topic_data[1], topic_data[2]
                suggestions.append(topic)
                related_topics.append({
                    'topic': topic,
                    'frequency': count,
                    'coverage_ratio': ratio
                })
        
        # Get topic categories (simple approach)
        topic_categories = []
        if suggestions:
            # Extract common words as categories
            all_words = []
            for topic in suggestions:
                all_words.extend(topic.split())
            
            word_counts = Counter(all_words)
            topic_categories = [word for word, count in word_counts.most_common(10) if count > 1]
        
        return TopicSuggestionResponse(
            dataset=dataset,
            suggestions=suggestions,
            related_topics=related_topics,
            topic_categories=topic_categories
        )
        
    except Exception as e:
        logger.error(f"Error in suggest_topics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    argsme_status = "loaded" if argsme_model else "not loaded"
    wikir_status = "loaded" if wikir_model else "not loaded"
    
    return {
        "status": "healthy",
        "models": {
            "argsme": argsme_status,
            "wikir": wikir_status
        },
        "service": "Topic Detection Service"
    }

@app.get("/model-info")
async def get_model_info(dataset: str = Query("argsme", description="Dataset to get info for: 'argsme' or 'wikir'")):
    """Get information about the loaded model"""
    topic_model = get_model(dataset)
    if not topic_model:
        raise HTTPException(status_code=500, detail=f"{dataset.upper()} topic model not loaded")
    
    try:
        info = {
            "dataset": dataset,
            "total_documents": topic_model.get('total_documents', 0),
            "total_keywords": topic_model.get('total_keywords_extracted', 0),
            "unique_keywords": topic_model.get('unique_keywords', 0),
            "global_topics_count": len(topic_model.get('global_topics', [])),
            "processing_time": topic_model.get('processing_time_seconds', 0),
            "dataset_type": topic_model.get('performance_metrics', {}).get('dataset_type', 'unknown'),
            "used_existing_tfidf": topic_model.get('performance_metrics', {}).get('used_existing_tfidf', False),
            "clustering_skipped": topic_model.get('performance_metrics', {}).get('clustering_skipped', False)
        }
        
        # Add top topics preview
        global_topics = topic_model.get('global_topics', [])
        info['top_topics_preview'] = []
        for topic_data in global_topics[:10]:
            if len(topic_data) >= 3:
                topic, count, ratio = topic_data[0], topic_data[1], topic_data[2]
                info['top_topics_preview'].append({
                    'topic': topic,
                    'frequency': count,
                    'coverage_ratio': ratio
                })
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

@app.get("/datasets")
async def get_datasets():
    """Get information about available datasets"""
    return {
        "available_datasets": [
            {
                "name": "argsme",
                "description": "Argumentative dataset for debate and argument analysis",
                "status": "loaded" if argsme_model else "not loaded",
                "documents": argsme_model.get('total_documents', 0) if argsme_model else 0
            },
            {
                "name": "wikir",
                "description": "Wikipedia-based information retrieval dataset",
                "status": "loaded" if wikir_model else "not loaded",
                "documents": wikir_model.get('total_documents', 0) if wikir_model else 0
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)