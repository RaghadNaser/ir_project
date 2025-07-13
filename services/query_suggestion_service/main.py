#!/usr/bin/env python3
"""
Smart Query Suggestion Service
Provides intelligent query suggestions using multiple methods:
1. Popular queries (most frequent)
2. Similar queries (semantic similarity)
3. Trending queries (recent patterns)
4. Related queries (topic-based)
5. Auto-complete suggestions
6. Query expansion
7. Spell correction
8. Category-based suggestions
"""

import os
import sqlite3
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import nltk
from nltk.corpus import wordnet
from difflib import get_close_matches
from collections import Counter, defaultdict
import pandas as pd
import re
import json
from datetime import datetime, timedelta
import pickle
import threading
from sentence_transformers import SentenceTransformer
import faiss
import logging
# from rapidfuzz import process as rapidfuzz_process, fuzz as rapidfuzz_fuzz

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Query Suggestion Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path
DB_PATH = "data/ir_database_combined.db"
USER_QUERIES_FILE = "data/vectors/user_queries.tsv"
USER_QUERIES_LOCK = threading.Lock()

class SuggestionRequest(BaseModel):
    query: str
    dataset: str
    method: str = "smart"  # [smart, popular, similar, trending, related, autocomplete, expansion, correction, category]
    top_k: int = 10
    include_metadata: bool = True
    user_id: Optional[str] = None

def save_user_query(query: str, dataset: str, user_id: Optional[str] = None):
    """Save user query to a TSV file (thread-safe)"""
    if not query.strip():
        return
    os.makedirs(os.path.dirname(USER_QUERIES_FILE), exist_ok=True)
    with USER_QUERIES_LOCK:
        with open(USER_QUERIES_FILE, "a", encoding="utf-8") as f:
            # Save with timestamp, dataset, and user_id for future use
            f.write(f"{dataset}\t{query.strip()}\t{datetime.utcnow().isoformat()}\t{user_id or ''}\n")

def load_user_queries(dataset: str, user_id: Optional[str] = None) -> Tuple[List[str], Counter]:
    """Load user queries for the dataset, with frequency. If user_id is provided, filter by user."""
    queries = []
    if os.path.exists(USER_QUERIES_FILE):
        with open(USER_QUERIES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    file_dataset, query, timestamp = parts[:3]
                    file_user_id = parts[3] if len(parts) > 3 else ""
                    if file_dataset == dataset:
                        if user_id is None or file_user_id == user_id:
                            queries.append(query)
    return queries, Counter(queries)

class SmartQuerySuggestionService:
    def __init__(self):
        self.queries_cache = {}
        self.vectorizers = {}
        self.query_embeddings = {}
        self.popularity_scores = {}
        self.category_mapping = {}
        self.trending_queries = {}
        self.query_frequency = {}
        self.similarity_matrix = {}
        # SBERT embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"[Warning] Could not load SBERT model: {e}")
            self.embedding_model = None
        self.sbert_query_embeddings = {}
        
    def load_queries(self, dataset: str, user_id: Optional[str] = None) -> List[Dict]:
        """Load user queries for the dataset, with frequency. If user_id is provided, filter by user."""
        queries, counts = load_user_queries(dataset, user_id)
        unique_queries = list(set(queries))
        result = []
        for uq in unique_queries:
            result.append({
                'id': f'user_{hash(uq)}',
                'title': uq,
                'description': '',
                'text': uq,
                'type': 'user',
                'count': counts[uq]
            })
        return result
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'
        }
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords
    
    def calculate_popularity_scores(self, dataset: str) -> Dict[int, float]:
        """Calculate popularity scores for queries based on frequency"""
        if dataset in self.popularity_scores:
            return self.popularity_scores[dataset]
        queries = self.load_queries(dataset)
        popularity = {}
        for query in queries:
            score = float(query.get('count', 1))  # عدد مرات التكرار هو المقياس الأساسي
            popularity[query['id']] = score
        self.popularity_scores[dataset] = popularity
        return popularity
    
    def is_valid_suggestion(self, text: str) -> bool:
        text = text.strip()
        if len(text) < 3:
            return False
        # استبعاد إذا كان النص كله أرقام أو رموز
        if not any(c.isalpha() for c in text):
            return False
        return True

    def get_popular_suggestions(self, dataset: str, top_k: int) -> List[Dict]:
        """Get popular queries"""
        queries = self.load_queries(dataset)
        popularity = self.calculate_popularity_scores(dataset)
        
        # Sort queries by popularity
        sorted_queries = sorted(queries, key=lambda x: popularity.get(x['id'], 0), reverse=True)
        
        suggestions = []
        for query in sorted_queries:
            if not self.is_valid_suggestion(query['title']):
                continue
            suggestions.append({
                'query': query['title'],
                'score': popularity.get(query['id'], 0),
                'type': 'popular',
                'metadata': {
                    'id': query['id'],
                    'description': query['description'],
                    'source': query['type']
                }
            })
        
        return suggestions[:top_k]
    
    def build_similarity_matrix(self, dataset: str):
        """Build similarity matrix between queries"""
        if dataset in self.similarity_matrix:
            return self.similarity_matrix[dataset]
        
        queries = self.load_queries(dataset)
        if not queries:
            return {}
        
        # Extract texts
        texts = [query['title'] for query in queries]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Transform texts to vectors
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Save results
        self.vectorizers[dataset] = vectorizer
        self.query_embeddings[dataset] = tfidf_matrix
        self.similarity_matrix[dataset] = similarity_matrix
        
        return similarity_matrix
    
    def get_similar_suggestions(self, query: str, dataset: str, top_k: int, user_id: Optional[str] = None) -> List[Dict]:
        """Get similar queries"""
        queries = self.load_queries(dataset, user_id)
        if not queries:
            # fallback to all users if user has no history
            queries = self.load_queries(dataset)
        if not queries:
            return []
        
        # Build similarity matrix if not exists
        similarity_matrix = self.build_similarity_matrix(dataset)
        
        # Transform input query
        vectorizer = self.vectorizers[dataset]
        query_vector = vectorizer.transform([query])
        
        # Calculate similarity with all queries
        similarities = cosine_similarity(query_vector, self.query_embeddings[dataset]).flatten()
        
        # Sort results
        top_indices = np.argsort(similarities)[::-1][:top_k*2]  # Take double for diversity
        
        suggestions = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Similarity threshold
                original_query = queries[idx]
                if not self.is_valid_suggestion(original_query['title']):
                    continue
                suggestions.append({
                    'query': original_query['title'],
                    'score': float(similarities[idx]),
                    'type': 'similar',
                    'metadata': {
                        'id': original_query['id'],
                        'similarity': float(similarities[idx]),
                        'source': original_query['type']
                    }
                })
        
        return suggestions[:top_k]
    
    def get_trending_suggestions(self, dataset: str, top_k: int) -> List[Dict]:
        """Get trending queries"""
        queries = self.load_queries(dataset)
        if not queries:
            return []
        
        # Extract keywords from recent queries
        recent_keywords = []
        for query in queries:
            keywords = self.extract_keywords(query['title'])
            recent_keywords.extend(keywords)
        
        # Count keyword frequency
        keyword_freq = Counter(recent_keywords)
        trending_keywords = [kw for kw, freq in keyword_freq.most_common(top_k * 2)]
        
        # Generate trending queries
        trending_queries = []
        
        for keyword in trending_keywords[:top_k]:
            trending_queries.append({
                'query': f"What is {keyword}?",
                'score': keyword_freq[keyword],
                'type': 'trending',
                'metadata': {
                    'keyword': keyword,
                    'frequency': keyword_freq[keyword],
                    'trend_score': keyword_freq[keyword] / max(keyword_freq.values())
                }
            })
        return trending_queries
    
    def get_related_suggestions(self, query: str, dataset: str, top_k: int, user_id: Optional[str] = None) -> List[Dict]:
        """Get related queries"""
        queries = self.load_queries(dataset, user_id)
        if not queries:
            queries = self.load_queries(dataset)
        if not queries:
            return []
        
        # Extract keywords from input query
        query_keywords = set(self.extract_keywords(query))
        
        related_suggestions = []
        for original_query in queries:
            # Extract keywords from original query
            original_keywords = set(self.extract_keywords(original_query['title']))
            
            # Calculate overlap
            overlap = len(query_keywords.intersection(original_keywords))
            if overlap > 0:
                related_score = overlap / max(len(query_keywords), len(original_keywords))
                if related_score > 0.2:  # Relatedness threshold
                    related_suggestions.append({
                        'query': original_query['title'],
                        'score': related_score,
                        'type': 'related',
                        'metadata': {
                            'id': original_query['id'],
                            'overlap': overlap,
                            'related_score': related_score,
                            'shared_keywords': list(query_keywords.intersection(original_keywords))
                        }
                    })
        
        # Sort by relatedness score
        related_suggestions.sort(key=lambda x: x['score'], reverse=True)
        return related_suggestions[:top_k]
    
    def get_autocomplete_suggestions(self, query: str, dataset: str, top_k: int) -> List[Dict]:
        """Get auto-complete suggestions with fuzzy matching"""
        queries = self.load_queries(dataset)
        if not queries or len(query) < 2:
            return []
        query_lower = query.lower().strip()
        autocomplete_suggestions = []
        titles = [original_query['title'] for original_query in queries]
        # 1. بادئة أو احتواء
        for original_query in queries:
            title_lower = original_query['title'].lower().strip()
            if title_lower.startswith(query_lower):
                if not self.is_valid_suggestion(original_query['title']):
                    continue
                autocomplete_suggestions.append({
                    'query': original_query['title'],
                    'score': 1.0,
                    'type': 'autocomplete',
                    'metadata': {
                        'id': original_query['id'],
                        'match_type': 'prefix',
                        'matched_part': query
                    }
                })
            elif query_lower in title_lower:
                if not self.is_valid_suggestion(original_query['title']):
                    continue
                autocomplete_suggestions.append({
                    'query': original_query['title'],
                    'score': 0.8,
                    'type': 'autocomplete',
                    'metadata': {
                        'id': original_query['id'],
                        'match_type': 'contains',
                        'matched_part': query
                    }
                })
        # 2. Fuzzy matching إذا لم يوجد اقتراح كافٍ
        # if len(autocomplete_suggestions) < top_k:
        #     fuzzy_matches = rapidfuzz_process.extract(
        #         query,
        #         titles,
        #         scorer=rapidfuzz_fuzz.QRatio,
        #         limit=top_k*2
        #     )
        #     for match_title, score, idx in fuzzy_matches:
        #         if score >= 70 and all(s['query'] != match_title for s in autocomplete_suggestions):
        #             if not self.is_valid_suggestion(match_title):
        #                 continue
        #             autocomplete_suggestions.append({
        #                 'query': match_title,
        #                 'score': score/100.0,
        #                 'type': 'fuzzy_autocomplete',
        #                 'metadata': {
        #                     'id': queries[idx]['id'],
        #                     'match_type': 'fuzzy',
        #                     'similarity': score/100.0
        #                 }
        #             })
        # ترتيب النتائج
        autocomplete_suggestions.sort(key=lambda x: x['score'], reverse=True)
        return autocomplete_suggestions[:top_k]
    
    def get_expansion_suggestions(self, query: str, dataset: str, top_k: int) -> List[Dict]:
        """Get expansion suggestions"""
        # Extract keywords
        keywords = self.extract_keywords(query)
        
        expansion_suggestions = []
        
        for keyword in keywords:
            # Search for synonyms
            synonyms = []
            for syn in wordnet.synsets(keyword):
                for lemma in syn.lemmas():
                    if lemma.name() != keyword:
                        synonyms.append(lemma.name())
            
            # Create expanded queries
            for synonym in synonyms[:3]:  # First 3 synonyms
                expanded_query = query.replace(keyword, synonym)
                expansion_suggestions.append({
                    'query': expanded_query,
                    'score': 0.7,
                    'type': 'expansion',
                    'metadata': {
                        'original_keyword': keyword,
                        'synonym': synonym,
                        'expansion_type': 'synonym'
                    }
                })
        
        return expansion_suggestions[:top_k]
    
    def get_correction_suggestions(self, query: str, dataset: str, top_k: int, user_id: Optional[str] = None) -> List[Dict]:
        """Get spell correction suggestions (احترافي)"""
        queries = self.load_queries(dataset, user_id)
        if not queries:
            queries = self.load_queries(dataset)
        if not queries:
            return []
        all_titles = [q['title'] for q in queries]
        # تصحيح تلقائي للكلمات (عربي/إنجليزي)
        words = query.split()
        corrected_words = []
        for word in words:
            if re.match(r'^[\u0600-\u06FF]+$', word) and self.spell_ar:
                # كلمة عربية
                corr = self.spell_ar.correction(word)
                corrected_words.append(corr if corr else word)
            else:
                # كلمة إنجليزية أو غير ذلك
                corr = self.spell_en.correction(word)
                corrected_words.append(corr if corr else word)
        corrected_query = ' '.join(corrected_words)
        suggestions = []
        if corrected_query != query:
            suggestions.append({
                'query': corrected_query,
                'score': 1.0,
                'type': 'correction',
                'metadata': {
                    'original_query': query,
                    'correction_type': 'spellchecker',
                    'lang': 'ar' if self.spell_ar else 'en'
                }
            })
        # إضافة الاقتراحات التقليدية (تشابه نصي)
        from difflib import get_close_matches
        close_matches = get_close_matches(query, all_titles, n=top_k, cutoff=0.6)
        for match in close_matches:
            if match != corrected_query:
                suggestions.append({
                    'query': match,
                    'score': 0.9,
                    'type': 'correction',
                    'metadata': {
                        'original_query': query,
                        'correction_type': 'close_match',
                        'similarity': 0.9
                    }
                })
        return suggestions[:top_k]
    
    def get_category_suggestions(self, query: str, dataset: str, top_k: int, user_id: Optional[str] = None) -> List[Dict]:
        """Get category-based suggestions"""
        # Determine query category
        query_lower = query.lower()
        
        categories = {
            'education': ['education', 'school', 'university', 'student', 'teacher', 'learning', 'study'],
            'health': ['health', 'medical', 'doctor', 'hospital', 'disease', 'treatment', 'medicine'],
            'technology': ['technology', 'computer', 'software', 'internet', 'digital', 'online'],
            'politics': ['government', 'politics', 'election', 'vote', 'law', 'policy'],
            'environment': ['environment', 'climate', 'pollution', 'energy', 'sustainability'],
            'economy': ['economy', 'money', 'finance', 'business', 'market', 'trade']
        }
        
        # Determine category
        detected_category = None
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_category = category
                break
        
        if not detected_category:
            return []
        
        # Search for queries in same category
        queries = self.load_queries(dataset, user_id)
        category_suggestions = []
        
        for original_query in queries:
            title_lower = original_query['title'].lower()
            if any(keyword in title_lower for keyword in categories[detected_category]):
                category_suggestions.append({
                    'query': original_query['title'],
                    'score': 0.8,
                    'type': 'category',
                    'metadata': {
                        'id': original_query['id'],
                        'category': detected_category,
                        'category_keywords': categories[detected_category]
                    }
                })
        
        return category_suggestions[:top_k]
    
    def get_smart_suggestions(self, query: str, dataset: str, top_k: int, user_id: Optional[str] = None) -> List[Dict]:
        """Get smart suggestions combining multiple methods, personalized if user_id is provided"""
        all_suggestions = []
        
        # 1. Personalized similar suggestions (30%)
        similar_suggestions = self.get_similar_suggestions(query, dataset, int(top_k * 0.3), user_id=user_id)
        all_suggestions.extend(similar_suggestions)
        
        # 2. Personalized related suggestions (25%)
        related_suggestions = self.get_related_suggestions(query, dataset, int(top_k * 0.25), user_id=user_id)
        all_suggestions.extend(related_suggestions)
        
        # 3. Personalized autocomplete suggestions (20%)
        autocomplete_suggestions = self.get_autocomplete_suggestions(query, dataset, int(top_k * 0.2), user_id=user_id)
        all_suggestions.extend(autocomplete_suggestions)
        
        # 4. Correction suggestions (15%)
        correction_suggestions = self.get_correction_suggestions(query, dataset, int(top_k * 0.15), user_id=user_id)
        all_suggestions.extend(correction_suggestions)
        
        # 5. Category suggestions (10%)
        category_suggestions = self.get_category_suggestions(query, dataset, int(top_k * 0.1), user_id=user_id)
        all_suggestions.extend(category_suggestions)
        
        # Remove duplicates
        seen_queries = set()
        unique_suggestions = []
        for suggestion in all_suggestions:
            if suggestion['query'] not in seen_queries:
                seen_queries.add(suggestion['query'])
                unique_suggestions.append(suggestion)
        
        # Sort by score
        unique_suggestions.sort(key=lambda x: x['score'], reverse=True)
        return unique_suggestions[:top_k]

    def get_semantic_suggestions(self, query: str, dataset: str, top_k: int, threshold: float = 0.3) -> list:
        """اقتراحات دلالية باستخدام Embedding (BERT/Sentence Transformers)"""
        queries = self.load_queries(dataset)
        if not queries or len(query) < 2:
            return []
        texts = [q['title'] for q in queries]
        # استخدم SBERT إذا متوفر
        if self.embedding_model is not None:
            # cache embeddings for dataset
            if dataset not in self.sbert_query_embeddings:
                self.sbert_query_embeddings[dataset] = self.embedding_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            embeddings = self.sbert_query_embeddings[dataset]
            # استخدم FAISS
            if not hasattr(self, 'faiss_indexes'):
                self.faiss_indexes = {}
            if dataset not in self.faiss_indexes:
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                faiss.normalize_L2(embeddings)
                index.add(embeddings)
                self.faiss_indexes[dataset] = index
            else:
                index = self.faiss_indexes[dataset]
            # استعلام embedding
            query_vec = self.embedding_model.encode([query], show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(query_vec)
            D, I = index.search(query_vec, top_k*2)  # top_k*2 for filtering
            suggestions = []
            for rank, (idx, score) in enumerate(zip(I[0], D[0])):
                if idx < 0 or score < threshold:
                    continue
                if not self.is_valid_suggestion(queries[idx]['title']):
                    continue
                suggestions.append({
                    'query': queries[idx]['title'],
                    'score': float(score),
                    'type': 'semantic',
                    'metadata': {
                        'id': queries[idx]['id'],
                        'similarity': float(score),
                        'source': queries[idx]['type']
                    }
                })
                if len(suggestions) >= top_k:
                    break
            return suggestions
        else:
            # fallback to TF-IDF
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(texts)
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            suggestions = []
            for idx, sim in enumerate(similarities):
                if sim > threshold:
                    if not self.is_valid_suggestion(queries[idx]['title']):
                        continue
                    suggestions.append({
                        'query': queries[idx]['title'],
                        'score': float(sim),
                        'type': 'semantic',
                        'metadata': {
                            'id': queries[idx]['id'],
                            'similarity': float(sim),
                            'source': queries[idx]['type']
                        }
                    })
            suggestions.sort(key=lambda x: x['score'], reverse=True)
            return suggestions[:top_k]

    def get_hybrid_suggestions(self, query: str, dataset: str, top_k: int) -> list:
        """دمج الاقتراحات الدلالية والشائعة والاستكمال"""
        semantic = self.get_semantic_suggestions(query, dataset, int(top_k * 0.5))
        popular = self.get_popular_suggestions(dataset, int(top_k * 0.3))
        autocomplete = self.get_autocomplete_suggestions(query, dataset, int(top_k * 0.2))
        # دمج بدون تكرار
        seen = set()
        results = []
        for group in [semantic, popular, autocomplete]:
            for s in group:
                if s['query'] not in seen:
                    seen.add(s['query'])
                    results.append(s)
        return results[:top_k]

    def extract_terms_from_documents(self, dataset: str, top_k_terms: int = 1000) -> List[Dict]:
        """Extract important terms from documents in the database"""
        try:
            # Connect to database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if database file exists
            if not os.path.exists(DB_PATH):
                logger.error(f"Database file not found: {DB_PATH}")
                return []
            
            # Check available tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            logger.info(f"Available tables: {table_names}")
            
            # Extract texts from database
            if dataset == "argsme":
                if "argsme_raw" not in table_names:
                    logger.error("argsme_raw table not found")
                    return []
                    
                sql = """
                SELECT doc_id, conclusion, premises_texts, source_title, topic
                FROM argsme_raw 
                LIMIT 1000
                """
            else:  # wikir
                if "wikir_docs" not in table_names:
                    logger.error("wikir_docs table not found")
                    return []
                    
                sql = """
                SELECT doc_id, text
                FROM wikir_docs
                LIMIT 1000
                """
            
            cursor.execute(sql)
            rows = cursor.fetchall()
            logger.info(f"Retrieved {len(rows)} documents from {dataset}")
            
            if not rows:
                logger.warning(f"No documents found in {dataset}")
                return []
            
            # Collect texts
            documents = []
            for row in rows:
                if dataset == "argsme":
                    # Combine conclusion and premises_texts
                    content_parts = []
                    if row[1]:  # conclusion
                        content_parts.append(str(row[1]))
                    if row[2]:  # premises_texts
                        content_parts.append(str(row[2]))
                    if row[3]:  # source_title
                        content_parts.append(str(row[3]))
                    
                    full_content = " ".join(content_parts)
                    if len(full_content.strip()) > 10:  # Only add if content is meaningful
                        documents.append({
                            'doc_id': str(row[0]),
                            'content': full_content,
                            'topic': str(row[4]) if row[4] else "Unknown"
                        })
                else:
                    content = str(row[1])
                    if len(content.strip()) > 10:  # Only add if content is meaningful
                        documents.append({
                            'doc_id': str(row[0]),
                            'content': content,
                            'topic': "WIKIR Document"
                        })
            
            logger.info(f"Processed {len(documents)} documents with meaningful content")
            
            if not documents:
                logger.warning("No documents with meaningful content found")
                return []
            
            # Extract keywords using TF-IDF
            texts = [doc['content'] for doc in documents]
            
            # Clean texts
            cleaned_texts = []
            for text in texts:
                # Simple cleaning
                text = re.sub(r'[^\w\s]', ' ', text.lower())
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > 20:  # Increased minimum length
                    cleaned_texts.append(text)
            
            logger.info(f"Cleaned {len(cleaned_texts)} texts")
            
            if not cleaned_texts:
                logger.warning("No cleaned texts available")
                return []
            
            # Use TF-IDF to extract keywords
            vectorizer = TfidfVectorizer(
                max_features=min(top_k_terms, 500),  # Limit for testing
                stop_words='english',
                ngram_range=(1, 2),  # Single and double words only
                min_df=1,  # Must appear in at least 1 document
                max_df=0.9  # Must not appear in more than 90% of documents
            )
            
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            logger.info(f"Extracted {len(feature_names)} features")
            
            # Calculate average TF-IDF for each term
            term_scores = {}
            for i, term in enumerate(feature_names):
                # Calculate average TF-IDF across all documents
                avg_score = tfidf_matrix[:, i].mean()
                if avg_score > 0.001:  # Lower threshold
                    term_scores[term] = avg_score
            
            logger.info(f"Found {len(term_scores)} terms with scores > 0.001")
            
            # Sort terms by importance
            sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Convert to list of dictionaries
            extracted_terms = []
            for term, score in sorted_terms[:top_k_terms]:
                extracted_terms.append({
                    'term': term,
                    'score': float(score),
                    'type': 'extracted_from_documents',
                    'dataset': dataset,
                    'frequency': int(score * 1000)  # Frequency estimate
                })
            
            logger.info(f"Returning {len(extracted_terms)} terms")
            return extracted_terms
            
        except Exception as e:
            logger.error(f"Error extracting terms from documents: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()

    def build_term_embeddings(self, dataset: str, terms: List[Dict]) -> Dict:
        """Build embedding vectors for extracted terms"""
        if not terms or not self.embedding_model:
            return {}
        
        try:
            # Extract texts only
            term_texts = [term['term'] for term in terms]
            
            # Convert to embeddings
            embeddings = self.embedding_model.encode(
                term_texts, 
                show_progress_bar=False, 
                convert_to_numpy=True
            )
            
            # Create FAISS index for fast search
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            
            # Normalize embeddings
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            
            # Save data
            term_embeddings_data = {
                'terms': terms,
                'embeddings': embeddings,
                'index': index,
                'term_to_idx': {term['term']: i for i, term in enumerate(terms)}
            }
            
            # Save in cache
            if not hasattr(self, 'term_embeddings_cache'):
                self.term_embeddings_cache = {}
            
            self.term_embeddings_cache[dataset] = term_embeddings_data
            
            return term_embeddings_data
            
        except Exception as e:
            logger.error(f"Error building term embeddings: {str(e)}")
            return {}

    def get_semantic_term_suggestions(self, query: str, dataset: str, top_k: int = 10, threshold: float = 0.3) -> List[Dict]:
        """Get semantic suggestions from terms extracted from documents"""
        
        # Check if term embeddings exist
        if not hasattr(self, 'term_embeddings_cache') or dataset not in self.term_embeddings_cache:
            # Extract terms and build embeddings if not available
            terms = self.extract_terms_from_documents(dataset)
            if not terms:
                return []
            
            term_data = self.build_term_embeddings(dataset, terms)
            if not term_data:
                return []
        
        term_data = self.term_embeddings_cache[dataset]
        
        try:
            # Convert query to embedding
            query_embedding = self.embedding_model.encode([query], show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            D, I = term_data['index'].search(query_embedding, top_k * 2)  # Double for filtering
            
            suggestions = []
            for rank, (idx, score) in enumerate(zip(I[0], D[0])):
                if idx < 0 or score < threshold:
                    continue
                
                term_info = term_data['terms'][idx]
                suggestions.append({
                    'query': term_info['term'],
                    'score': float(score),
                    'type': 'semantic_term',
                    'metadata': {
                        'source': 'document_extraction',
                        'dataset': dataset,
                        'term_score': term_info['score'],
                        'frequency': term_info['frequency'],
                        'semantic_similarity': float(score)
                    }
                })
                
                if len(suggestions) >= top_k:
                    break
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting semantic term suggestions: {str(e)}")
            return []

    def get_hybrid_term_suggestions(self, query: str, dataset: str, top_k: int = 10) -> List[Dict]:
        """Combine semantic term suggestions with user suggestions"""
        
        # 1. Term suggestions from documents (60%)
        semantic_terms = self.get_semantic_term_suggestions(query, dataset, int(top_k * 0.6))
        
        # 2. Previous user suggestions (40%)
        user_suggestions = self.get_semantic_suggestions(query, dataset, int(top_k * 0.4))
        
        # Merge results
        all_suggestions = []
        seen_queries = set()
        
        # Add term suggestions first
        for suggestion in semantic_terms:
            if suggestion['query'] not in seen_queries:
                seen_queries.add(suggestion['query'])
                all_suggestions.append(suggestion)
        
        # Add user suggestions
        for suggestion in user_suggestions:
            if suggestion['query'] not in seen_queries:
                seen_queries.add(suggestion['query'])
                all_suggestions.append(suggestion)
        
        # Sort by score
        all_suggestions.sort(key=lambda x: x['score'], reverse=True)
        return all_suggestions[:top_k]


# Create service instance
suggestion_service = SmartQuerySuggestionService()

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Starting Query Suggestion Service...")
    logger.info("Service initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Query Suggestion Service...")
    # Cleanup any resources if needed

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "query_suggestion_service",
        "version": "1.0.0"
    }

@app.post("/suggest")
def suggest(req: SuggestionRequest):
    save_user_query(req.query, req.dataset)
    suggestions = []
    
    if req.method == "hybrid":
        suggestions = suggestion_service.get_hybrid_suggestions(req.query, req.dataset, req.top_k)
    elif req.method == "semantic":
        suggestions = suggestion_service.get_semantic_suggestions(req.query, req.dataset, req.top_k)
    elif req.method == "popular":
        suggestions = suggestion_service.get_popular_suggestions(req.dataset, req.top_k)
    elif req.method == "similar":
        suggestions = suggestion_service.get_similar_suggestions(req.query, req.dataset, req.top_k)
    elif req.method == "trending":
        suggestions = suggestion_service.get_trending_suggestions(req.dataset, req.top_k)
    elif req.method == "related":
        suggestions = suggestion_service.get_related_suggestions(req.query, req.dataset, req.top_k)
    elif req.method == "autocomplete":
        suggestions = suggestion_service.get_autocomplete_suggestions(req.query, req.dataset, req.top_k)
    elif req.method == "expansion":
        suggestions = suggestion_service.get_expansion_suggestions(req.query, req.dataset, req.top_k)
    elif req.method == "correction":
        suggestions = suggestion_service.get_correction_suggestions(req.query, req.dataset, req.top_k)
    elif req.method == "category":
        suggestions = suggestion_service.get_category_suggestions(req.query, req.dataset, req.top_k)
    elif req.method == "semantic_terms":
        suggestions = suggestion_service.get_semantic_term_suggestions(req.query, req.dataset, req.top_k)
    elif req.method == "hybrid_terms":
        suggestions = suggestion_service.get_hybrid_term_suggestions(req.query, req.dataset, req.top_k)
    
    # Remove metadata if not requested
    if not req.include_metadata:
        for suggestion in suggestions:
            suggestion.pop('metadata', None)
    
    return {
        "suggestions": suggestions,
        "method": req.method,
        "dataset": req.dataset,
        "query": req.query,
        "count": len(suggestions)
    }

@app.get("/methods")
def get_available_methods():
    """Get available methods (user-driven only)"""
    return {
        "methods": [
            {
                "name": "hybrid",
                "description": "Hybrid: Semantic + Popular + Autocomplete",
                "best_for": "Most effective and smart suggestions"
            },
            {
                "name": "semantic",
                "description": "Semantic (Embedding-based) Suggestions",
                "best_for": "Smart meaning-based suggestions"
            },
            {
                "name": "popular",
                "description": "Most Popular User Queries",
                "best_for": "Discovering common user searches"
            },
            {
                "name": "autocomplete",
                "description": "Auto-complete from User Queries",
                "best_for": "Completing user search queries"
            },
            {
                "name": "semantic_terms",
                "description": "Semantic Terms from Documents",
                "best_for": "Finding relevant terms from document content"
            },
            {
                "name": "hybrid_terms",
                "description": "Hybrid Terms: Document Terms + User Queries",
                "best_for": "Best of both worlds - document terms and user patterns"
            }
        ]
    }

@app.post("/extract-terms")
async def extract_terms(request: Request):
    """Extract terms from documents"""
    try:
        data = await request.json()
        dataset = data.get("dataset", "argsme")
        top_k = data.get("top_k", 1000)
        
        logger.info(f"Extracting terms for dataset: {dataset}, top_k: {top_k}")
        terms = suggestion_service.extract_terms_from_documents(dataset, top_k)
        logger.info(f"Successfully extracted {len(terms)} terms")
        return {
            "dataset": dataset,
            "terms_count": len(terms),
            "terms": terms[:100],  # Return first 100 terms only for display
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in extract_terms endpoint: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "dataset": data.get("dataset", "argsme") if 'data' in locals() else "unknown",
            "terms_count": 0,
            "terms": [],
            "status": "error",
            "error": str(e)
        }

@app.post("/build-term-embeddings")
async def build_term_embeddings(request: Request):
    """Build embedding vectors for terms"""
    try:
        data = await request.json()
        dataset = data.get("dataset", "argsme")
        top_k = data.get("top_k", 1000)
        
        # Extract terms first
        terms = suggestion_service.extract_terms_from_documents(dataset, top_k)
        if not terms:
            return {
                "dataset": dataset,
                "status": "error",
                "error": "No terms extracted"
            }
        
        # Build embeddings
        term_data = suggestion_service.build_term_embeddings(dataset, terms)
        if not term_data:
            return {
                "dataset": dataset,
                "status": "error",
                "error": "Failed to build embeddings"
            }
        
        return {
            "dataset": dataset,
            "terms_count": len(terms),
            "embeddings_shape": term_data['embeddings'].shape,
            "status": "success",
            "message": f"Built embeddings for {len(terms)} terms"
        }
    except Exception as e:
        return {
            "dataset": data.get("dataset", "argsme") if 'data' in locals() else "unknown",
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8010,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1) 