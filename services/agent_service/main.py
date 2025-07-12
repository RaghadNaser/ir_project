#!/usr/bin/env python3
"""
Professional AI Agent Service with Advanced Intelligence
Features:
1. Multi-step reasoning with context awareness
2. Advanced query understanding and refinement
3. Intelligent response synthesis
4. Multi-language support (Arabic/English)
5. Session management with long-term memory
6. Real-time conversation analysis
7. Smart suggestions and recommendations
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import requests
import asyncio
import json
import time
import logging
from datetime import datetime
from collections import deque
import re
from dataclasses import dataclass
from enum import Enum
import sqlite3
import hashlib
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_global_embedding_model = None
_global_agent_instance = None
_model_lock = threading.Lock()
_model_loaded = False
_agent_initialized = False

def get_agent_instance():
    """Get or create the global agent instance with proper caching"""
    global _global_agent_instance, _agent_initialized
    
    if _global_agent_instance is None or not _agent_initialized:
        with _model_lock:
            if _global_agent_instance is None or not _agent_initialized:
                logger.info("Creating new agent instance...")
                _global_agent_instance = ProfessionalAIAgent()
                _agent_initialized = True
                logger.info("Agent instance created and cached successfully")
    
    return _global_agent_instance

app = FastAPI(
    title="Professional AI Agent Service",
    description="Advanced intelligent agent with multi-language support, context awareness, and long-term memory",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
SERVICE_URLS = {
    "hybrid": "http://localhost:8005",
    "topic_detection": "http://localhost:8006", 
    "query_suggestion": "http://localhost:8010",
    "preprocessing": "http://localhost:8002"
}

class ConversationState(Enum):
    INITIAL = "initial"
    UNDERSTANDING = "understanding"
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    REFINING = "refining"
    SYNTHESIZING = "synthesizing"
    LEARNING = "learning"
    COMPLETE = "complete"

@dataclass
class ConversationContext:
    """Advanced conversation context with long-term memory and learning capabilities"""
    session_id: str
    user_id: Optional[str] = None
    dataset: str = "argsme"
    conversation_history: deque = None
    current_state: ConversationState = ConversationState.INITIAL
    search_results: List[Dict] = None
    topics: List[Dict] = None
    suggestions: List[str] = None
    refined_queries: List[str] = None
    user_preferences: Dict[str, Any] = None
    language_detected: str = "en"
    confidence_scores: List[float] = None
    learning_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = deque(maxlen=20)  # Increased memory
        if self.search_results is None:
            self.search_results = []
        if self.topics is None:
            self.topics = []
        if self.suggestions is None:
            self.suggestions = []
        if self.refined_queries is None:
            self.refined_queries = []
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.confidence_scores is None:
            self.confidence_scores = []
        if self.learning_data is None:
            self.learning_data = {}

class AgentRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    dataset: str = "argsme"
    conversation_mode: str = "interactive"  # "interactive", "direct", "analytical"
    max_results: int = 10
    include_topics: bool = True
    include_suggestions: bool = True

class AgentResponse(BaseModel):
    response: str
    session_id: str
    conversation_state: str
    search_results: List[Dict] = []
    topics: List[Dict] = []
    suggestions: List[str] = []
    refined_queries: List[str] = []
    confidence_score: float
    reasoning_steps: List[str] = []
    execution_time: float

class ConversationRequest(BaseModel):
    session_id: str
    message: str
    dataset: str = "argsme"

class ConversationResponse(BaseModel):
    session_id: str
    response: str
    context: Dict[str, Any]
    suggestions: List[str] = []

class ProfessionalAIAgent:
    """Professional AI Agent with advanced intelligence and learning capabilities"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProfessionalAIAgent, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("Initializing Professional AI Agent...")
            self.conversations: Dict[str, ConversationContext] = {}
            self.session_counter = 0
            self.language_model = None
            self.embedding_model = None
            self.memory_db = None
            self._initialize_ai_models()
            self._initialize_memory_database()
            ProfessionalAIAgent._initialized = True
            logger.info("Professional AI Agent initialization completed")
    
    def _initialize_ai_models(self):
        """Initialize AI models for language understanding and generation"""
        global _global_embedding_model, _model_loaded
        
        try:
            # Check if model is already loaded globally
            if _global_embedding_model is not None and _model_loaded:
                self.embedding_model = _global_embedding_model
                logger.info("Using cached AI model, skipping reload")
                return
                
            # Check if model is already loaded locally
            if hasattr(self, 'embedding_model') and self.embedding_model is not None:
                logger.info("AI models already initialized locally, skipping reload")
                return
                
            # Load model with thread safety
            with _model_lock:
                if _global_embedding_model is None or not _model_loaded:
                    logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2")
                    _global_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    _model_loaded = True
                    logger.info("AI models initialized successfully")
                
                self.embedding_model = _global_embedding_model
                
        except Exception as e:
            logger.warning(f"Could not initialize AI models: {e}")
            self.embedding_model = None
    
    def _initialize_memory_database(self):
        """Initialize SQLite database for long-term memory"""
        try:
            # Check if database is already initialized
            if hasattr(self, 'memory_db') and self.memory_db is not None:
                logger.info("Memory database already initialized, skipping reload")
                return
                
            logger.info("Initializing memory database...")
            self.memory_db = sqlite3.connect(':memory:', check_same_thread=False)
            cursor = self.memory_db.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_memory (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT,
                    conversation_patterns TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS session_data (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    context_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.memory_db.commit()
            logger.info("Memory database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory database: {e}")
            self.memory_db = None
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        self.session_counter += 1
        return f"agent_session_{self.session_counter}_{int(time.time())}"
    
    def _get_or_create_context(self, session_id: str, dataset: str = "argsme") -> ConversationContext:
        """Get existing context or create new one"""
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationContext(
                session_id=session_id,
                dataset=dataset
            )
        return self.conversations[session_id]
    
    def _analyze_intent(self, message: str) -> Dict[str, Any]:
        """Advanced intent analysis with semantic understanding"""
        message_lower = message.lower()
        
        # Enhanced intent patterns with semantic variations
        patterns = {
            "search": ["find", "search", "look for", "show me", "get", "retrieve", "find me", "locate", "discover"],
            "explain": ["explain", "what is", "how does", "tell me about", "describe", "define", "clarify", "elaborate"],
            "compare": ["compare", "difference", "versus", "vs", "contrast", "similarities", "differences"],
            "analyze": ["analyze", "analysis", "study", "examine", "investigate", "research", "explore"],
            "refine": ["refine", "improve", "better results", "enhance", "optimize", "filter"],
            "clarify": ["clarify", "more specific", "details", "elaborate", "expand", "specify"],
            "summarize": ["summarize", "summary", "overview", "brief", "recap", "conclude"],
            "recommend": ["recommend", "suggest", "advise", "propose", "recommendation"],
            "evaluate": ["evaluate", "assess", "judge", "rate", "review", "critique"]
        }
        
        # Semantic similarity analysis if embedding model is available
        if self.embedding_model:
            try:
                message_embedding = self.embedding_model.encode([message])
                
                # Compare with intent keywords
                intent_scores = {}
                for intent, keywords in patterns.items():
                    keyword_embeddings = self.embedding_model.encode(keywords)
                    similarities = np.dot(keyword_embeddings, message_embedding.T).flatten()
                    intent_scores[intent] = np.max(similarities)
                
                # Get top intents based on semantic similarity
                sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
                detected_intents = [intent for intent, score in sorted_intents if score > 0.3]
                
                if detected_intents:
                    confidence = float(sorted_intents[0][1])
                    return {
                        "primary_intent": detected_intents[0],
                        "all_intents": detected_intents[:3],
                        "confidence": min(confidence, 0.95),
                        "semantic_analysis": True
                    }
            except Exception as e:
                logger.warning(f"Semantic analysis failed: {e}")
        
        # Fallback to keyword-based analysis
        detected_intents = []
        for intent, keywords in patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_intents.append(intent)
        
        # Default to search if no specific intent detected
        if not detected_intents:
            detected_intents = ["search"]
        
        return {
            "primary_intent": detected_intents[0],
            "all_intents": detected_intents,
            "confidence": 0.8 if len(detected_intents) == 1 else 0.6,
            "semantic_analysis": False
        }
    
    def _detect_language(self, message: str) -> str:
        """Detect language of the message (Arabic/English)"""
        # Arabic character detection
        arabic_chars = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', message)
        arabic_ratio = len(arabic_chars) / len(message) if message else 0
        
        if arabic_ratio > 0.1:  # More than 10% Arabic characters
            return "ar"
        return "en"
    
    def _extract_keywords(self, message: str, language: str = "en") -> List[str]:
        """Extract important keywords from message with language-specific processing"""
        # Language-specific stop words
        stop_words = {
            "en": {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"},
            "ar": {"في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "تلك", "التي", "الذي", "الذين", "كان", "كانت", "يكون", "تكون", "أنا", "أنت", "هو", "هي", "نحن", "أنتم", "هم", "هن", "لي", "له", "لها", "لنا", "لكم", "لهم"}
        }
        
        # Extract words based on language
        if language == "ar":
            # Arabic word extraction
            words = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+', message)
        else:
            # English word extraction
            words = re.findall(r'\b\w+\b', message.lower())
        
        # Filter stop words
        current_stop_words = stop_words.get(language, stop_words["en"])
        keywords = [word for word in words if word not in current_stop_words and len(word) > 2]
        
        # Use semantic ranking if embedding model is available
        if self.embedding_model and keywords:
            try:
                keyword_embeddings = self.embedding_model.encode(keywords)
                # Calculate importance based on embedding diversity
                similarities = np.dot(keyword_embeddings, keyword_embeddings.T)
                importance_scores = 1 - np.mean(similarities, axis=1)
                ranked_keywords = [kw for _, kw in sorted(zip(importance_scores, keywords), reverse=True)]
                return ranked_keywords[:10]
            except Exception as e:
                logger.warning(f"Semantic keyword ranking failed: {e}")
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def _save_to_memory(self, user_id: str, context: ConversationContext):
        """Save conversation context to long-term memory"""
        if not self.memory_db or not user_id:
            return
        
        try:
            cursor = self.memory_db.cursor()
            
            # Save user preferences and patterns
            preferences = json.dumps(context.user_preferences)
            patterns = json.dumps({
                "frequent_topics": [topic.get('topic_name', '') for topic in context.topics[:3]],
                "preferred_language": context.language_detected,
                "avg_confidence": np.mean(context.confidence_scores) if context.confidence_scores else 0.5
            })
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_memory 
                (user_id, preferences, conversation_patterns, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, preferences, patterns))
            
            # Save session data
            session_data = json.dumps({
                "conversation_history": list(context.conversation_history),
                "search_results": context.search_results[:5],
                "topics": context.topics[:3]
            })
            
            cursor.execute('''
                INSERT OR REPLACE INTO session_data 
                (session_id, user_id, context_data)
                VALUES (?, ?, ?)
            ''', (context.session_id, user_id, session_data))
            
            self.memory_db.commit()
            
        except Exception as e:
            logger.error(f"Failed to save to memory: {e}")
    
    def _load_from_memory(self, user_id: str) -> Dict[str, Any]:
        """Load user preferences and patterns from memory"""
        if not self.memory_db or not user_id:
            return {}
        
        try:
            cursor = self.memory_db.cursor()
            cursor.execute('SELECT preferences, conversation_patterns FROM user_memory WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            
            if result:
                return {
                    "preferences": json.loads(result[0]) if result[0] else {},
                    "patterns": json.loads(result[1]) if result[1] else {}
                }
        except Exception as e:
            logger.error(f"Failed to load from memory: {e}")
        
        return {}
    
    def _refine_query(self, original_query: str, context: ConversationContext) -> List[str]:
        """Generate refined queries based on context, intent, and memory"""
        refined_queries = [original_query]
        
        # Extract keywords with language detection
        language = context.language_detected
        keywords = self._extract_keywords(original_query, language)
        
        # Generate variations
        if len(keywords) >= 2:
            # Create keyword combinations
            for i in range(len(keywords) - 1):
                for j in range(i + 1, min(i + 3, len(keywords))):
                    combination = " ".join(keywords[i:j+1])
                    if combination != original_query:
                        refined_queries.append(combination)
        
        # Add context-based refinements
        if context.topics:
            for topic in context.topics[:2]:  # Top 2 topics
                topic_name = topic.get('topic_name', '')
                if topic_name and topic_name not in original_query:
                    refined_queries.append(f"{original_query} {topic_name}")
        
        # Add memory-based refinements
        if context.user_id:
            memory_data = self._load_from_memory(context.user_id)
            if memory_data.get("patterns", {}).get("frequent_topics"):
                for topic in memory_data["patterns"]["frequent_topics"][:2]:
                    if topic and topic not in original_query:
                        refined_queries.append(f"{original_query} {topic}")
        
        return list(set(refined_queries))[:5]  # Remove duplicates and limit
    
    async def _call_service(self, service_name: str, endpoint: str, data: Dict) -> Dict:
        """Call external service with error handling"""
        try:
            service_url = SERVICE_URLS.get(service_name)
            if not service_url:
                raise Exception(f"Service {service_name} not found")
            
            url = f"{service_url}/{endpoint}"
            response = requests.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Service {service_name} returned {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error calling {service_name}: {e}")
            return {}
    
    async def _perform_search(self, query: str, dataset: str, max_results: int) -> List[Dict]:
        """Perform search using hybrid service"""
        try:
            search_data = {
                "dataset": dataset,
                "query": query,
                "top_k": max_results,
                "method": "fusion"
            }
            
            result = await self._call_service("hybrid", "search", search_data)
            return result.get("results", [])
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    async def _detect_topics(self, query: str, dataset: str) -> List[Dict]:
        """Detect topics using topic detection service"""
        try:
            topic_data = {
                "query": query,
                "dataset": dataset,
                "max_topics": 5,
                "min_relevance_score": 0.1
            }
            
            result = await self._call_service("topic_detection", "detect_topics", topic_data)
            return result.get("topics", [])
            
        except Exception as e:
            logger.error(f"Topic detection error: {e}")
            return []
    
    async def _get_suggestions(self, query: str, dataset: str) -> List[str]:
        """Get query suggestions"""
        try:
            suggestion_data = {
                "query": query,
                "dataset": dataset,
                "method": "semantic",
                "count": 5
            }
            
            result = await self._call_service("query_suggestion", "suggest", suggestion_data)
            suggestions = result.get("suggestions", [])
            return [s.get("query", "") for s in suggestions if s.get("query")]
            
        except Exception as e:
            logger.error(f"Suggestion error: {e}")
            return []
    
    async def _read_document_content(self, doc_id: str, dataset: str) -> Dict[str, Any]:
        """Read document content from database"""
        try:
            import sqlite3
            db_path = "data/ir_database_combined.db"
            
            if not os.path.exists(db_path):
                return {"error": "Database not found"}
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            if dataset.lower() == "argsme":
                cursor.execute("""
                    SELECT doc_id, conclusion, premises_texts, source_title, topic, acquisition
                    FROM argsme_raw 
                    WHERE doc_id = ?
                """, (doc_id,))
                row = cursor.fetchone()
                if row:
                    return {
                        "doc_id": doc_id,
                        "conclusion": row[1] if row[1] else "",
                        "premises_texts": row[2] if row[2] else "",
                        "source_title": row[3] if row[3] else "",
                        "topic": row[4] if row[4] else "",
                        "acquisition": row[5] if row[5] else "",
                        "full_text": f"{row[1] if row[1] else ''} {row[2] if row[2] else ''}".strip(),
                        "type": "argument"
                    }
            
            elif dataset.lower() == "wikir":
                cursor.execute("""
                    SELECT doc_id, text 
                    FROM wikir_docs 
                    WHERE doc_id = ?
                """, (doc_id,))
                row = cursor.fetchone()
                if row:
                    return {
                        "doc_id": doc_id,
                        "text": row[1] if row[1] else "",
                        "full_text": row[1] if row[1] else "",
                        "type": "wiki"
                    }
            
            conn.close()
            return {"error": f"Document {doc_id} not found"}
            
        except Exception as e:
            logger.error(f"Error reading document {doc_id}: {e}")
            return {"error": str(e)}
    
    async def _analyze_document_content(self, doc_content: Dict[str, Any], query: str) -> str:
        """Analyze document content and provide insights"""
        try:
            if "error" in doc_content:
                return f"❌ Error: {doc_content['error']}"
            
            full_text = doc_content.get("full_text", "")
            if not full_text:
                return "❌ No content available for analysis"
            
            # Extract key information
            analysis = []
            
            # Document type specific analysis
            if doc_content.get("type") == "argument":
                conclusion = doc_content.get("conclusion", "")
                premises = doc_content.get("premises_texts", "")
                topic = doc_content.get("topic", "")
                source = doc_content.get("source_title", "")
                
                analysis.append(f"📋 **Document Type:** Argument")
                if topic:
                    analysis.append(f"🏷️ **Topic:** {topic}")
                if source:
                    analysis.append(f"📚 **Source:** {source}")
                if conclusion:
                    analysis.append(f"💡 **Conclusion:** {conclusion[:200]}{'...' if len(conclusion) > 200 else ''}")
                if premises:
                    analysis.append(f"🔍 **Premises:** {premises[:200]}{'...' if len(premises) > 200 else ''}")
            
            elif doc_content.get("type") == "wiki":
                analysis.append(f"📋 **Document Type:** Wiki Article")
                analysis.append(f"📝 **Content Preview:** {full_text[:300]}{'...' if len(full_text) > 300 else ''}")
            
            # Relevance analysis using semantic similarity if available
            if self.embedding_model and full_text:
                try:
                    query_embedding = self.embedding_model.encode([query])
                    doc_embedding = self.embedding_model.encode([full_text[:1000]])  # Limit for performance
                    similarity = np.dot(query_embedding, doc_embedding.T)[0][0]
                    
                    if similarity > 0.7:
                        relevance = "🔴 High"
                    elif similarity > 0.5:
                        relevance = "🟡 Medium"
                    else:
                        relevance = "🟢 Low"
                    
                    analysis.append(f"🎯 **Relevance Score:** {similarity:.2f} ({relevance})")
                except Exception as e:
                    logger.warning(f"Semantic analysis failed: {e}")
            
            # Content length
            word_count = len(full_text.split())
            analysis.append(f"📊 **Word Count:** {word_count}")
            
            return "\n".join(analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return f"❌ Analysis error: {str(e)}"
    
    async def _get_document_summary(self, doc_content: Dict[str, Any]) -> str:
        """Generate a concise summary of the document"""
        try:
            if "error" in doc_content:
                return f"❌ Error: {doc_content['error']}"
            
            full_text = doc_content.get("full_text", "")
            if not full_text:
                return "❌ No content available for summary"
            
            # Simple summary based on content type
            if doc_content.get("type") == "argument":
                conclusion = doc_content.get("conclusion", "")
                if conclusion:
                    return f"💡 **Summary:** This argument concludes that {conclusion[:150]}{'...' if len(conclusion) > 150 else ''}"
                else:
                    return f"💡 **Summary:** Argument document with {len(full_text.split())} words"
            
            elif doc_content.get("type") == "wiki":
                # Extract first sentence or first 100 characters
                first_sentence = full_text.split('.')[0] if '.' in full_text else full_text[:100]
                return f"💡 **Summary:** {first_sentence}{'...' if len(first_sentence) >= 100 else ''}"
            
            return f"💡 **Summary:** Document with {len(full_text.split())} words"
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"❌ Summary error: {str(e)}"
    
    async def _extract_relevant_text(self, doc_content: Dict[str, Any], query: str) -> str:
        """Extract relevant text from document based on query"""
        try:
            if "error" in doc_content:
                return f"❌ Error: {doc_content['error']}"
            
            full_text = doc_content.get("full_text", "")
            if not full_text:
                return "❌ No content available for extraction"
            
            # Extract query keywords
            query_words = set(query.lower().split())
            
            # Document type specific extraction
            if doc_content.get("type") == "argument":
                conclusion = doc_content.get("conclusion", "")
                premises = doc_content.get("premises_texts", "")
                
                extracted_parts = []
                
                # Always show conclusion (first 300 characters)
                if conclusion:
                    conclusion_lower = conclusion.lower()
                    if any(word in conclusion_lower for word in query_words if len(word) > 2):
                        # Show full conclusion if it contains query keywords
                        extracted_parts.append(f"**Conclusion:** {conclusion}")
                    else:
                        # Show first 300 characters of conclusion
                        extracted_parts.append(f"**Conclusion:** {conclusion[:300]}{'...' if len(conclusion) > 300 else ''}")
                
                # Always show premises (first 500 characters)
                if premises:
                    premises_lower = premises.lower()
                    if any(word in premises_lower for word in query_words if len(word) > 2):
                        # Find relevant sentences
                        premises_sentences = premises.split('.')
                        relevant_sentences = []
                        
                        for sentence in premises_sentences:
                            sentence = sentence.strip()
                            if len(sentence) > 10:  # Only meaningful sentences
                                sentence_lower = sentence.lower()
                                if any(word in sentence_lower for word in query_words if len(word) > 2):
                                    relevant_sentences.append(sentence)
                        
                        if relevant_sentences:
                            # Show up to 4 relevant sentences
                            relevant_text = '. '.join(relevant_sentences[:4])
                            extracted_parts.append(f"**Relevant Premises:** {relevant_text}")
                        else:
                            # Show first 500 characters of premises
                            extracted_parts.append(f"**Premises:** {premises[:500]}{'...' if len(premises) > 500 else ''}")
                    else:
                        # Show first 500 characters of premises
                        extracted_parts.append(f"**Premises:** {premises[:500]}{'...' if len(premises) > 500 else ''}")
                
                return '\n\n'.join(extracted_parts)
            
            elif doc_content.get("type") == "wiki":
                # For wiki documents, always show content
                if any(word in full_text.lower() for word in query_words if len(word) > 2):
                    # Find sentences containing query keywords
                    sentences = full_text.split('.')
                    relevant_sentences = []
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) > 10:  # Only meaningful sentences
                            sentence_lower = sentence.lower()
                            if any(word in sentence_lower for word in query_words if len(word) > 2):
                                relevant_sentences.append(sentence)
                    
                    if relevant_sentences:
                        # Show up to 6 relevant sentences
                        relevant_text = '. '.join(relevant_sentences[:6])
                        return f"**Relevant Content:** {relevant_text}"
                    else:
                        # Show first 600 characters
                        return f"**Content Preview:** {full_text[:600]}{'...' if len(full_text) > 600 else ''}"
                else:
                    # Show first 600 characters if no relevant sentences found
                    return f"**Content Preview:** {full_text[:600]}{'...' if len(full_text) > 600 else ''}"
            
            # Generic extraction for other document types
            if any(word in full_text.lower() for word in query_words if len(word) > 2):
                # Find sentences containing query keywords
                sentences = full_text.split('.')
                relevant_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 10:
                        sentence_lower = sentence.lower()
                        if any(word in sentence_lower for word in query_words if len(word) > 2):
                            relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    relevant_text = '. '.join(relevant_sentences[:5])
                    return f"**Relevant Content:** {relevant_text}"
                else:
                    return f"**Content Preview:** {full_text[:500]}{'...' if len(full_text) > 500 else ''}"
            else:
                return f"**Content Preview:** {full_text[:500]}{'...' if len(full_text) > 500 else ''}"
            
        except Exception as e:
            logger.error(f"Error extracting relevant text: {e}")
            return f"❌ Extraction error: {str(e)}"
    
    async def _synthesize_response(self, context: ConversationContext, intent: Dict, original_query: str) -> str:
        """Synthesize intelligent response based on context and intent"""
        primary_intent = intent["primary_intent"]
        
        # If we have search results, show detailed analysis for most intents
        if context.search_results:
            if primary_intent == "refine":
                return await self._format_refinement_response(context, original_query)
            else:
                # For all other intents (search, explain, compare, analyze, clarify, summarize, etc.)
                # Show detailed search response with document analysis
                return await self._format_search_response(context, original_query)
        else:
            # No search results
            if primary_intent == "search":
                return "Sorry, I couldn't find any matching results for your search. Please try different keywords."
            elif primary_intent == "explain":
                return await self._format_explanation_response(context, original_query)
            elif primary_intent == "refine":
                return await self._format_refinement_response(context, original_query)
            else:
                return "No results found to analyze. Please try a different search query."
    
    async def _format_search_response(self, context: ConversationContext, original_query: str) -> str:
        """Format search results response with document content analysis and extracted text"""
        if not context.search_results:
            return "No matching results found for your search."
        
        response = f"Found {len(context.search_results)} results related to your query:\n\n"
        
        # Show search method information
        if context.search_results:
            first_result = context.search_results[0]
            sources = first_result.get("sources", [])
            fusion_method = first_result.get("fusion_method", "unknown")
            response += f"**🔍 Search Method:** Hybrid search using {', '.join(sources)} with {fusion_method} fusion\n\n"
        
        # Analyze top 3 results in detail with extracted text
        detailed_results = []
        for i, result in enumerate(context.search_results[:3], 1):
            doc_id = result.get("doc_id", "unknown")
            score = result.get("score", 0)
            title = result.get("title", f"Document {doc_id}")
            
            # Read document content
            doc_content = await self._read_document_content(doc_id, context.dataset)
            
            # Analyze document content
            analysis = await self._analyze_document_content(doc_content, original_query)
            summary = await self._get_document_summary(doc_content)
            
            # Extract relevant text from document
            extracted_text = await self._extract_relevant_text(doc_content, original_query)
            
            detailed_results.append({
                "doc_id": doc_id,
                "title": title,
                "score": score,
                "analysis": analysis,
                "summary": summary,
                "extracted_text": extracted_text,
                "content": doc_content
            })
        
        # Format detailed response with extracted text
        for i, result in enumerate(detailed_results, 1):
            # Get source information from original result
            original_result = next((r for r in context.search_results if r.get("doc_id") == result['doc_id']), {})
            sources = original_result.get("sources", [])
            fusion_method = original_result.get("fusion_method", "unknown")
            
            response += f"## 📄 **{result['title']}** (Document {result['doc_id']}) (confidence: {result['score']:.2f})\n\n"
            response += f"**🔍 Sources:** {', '.join(sources)} | **Fusion:** {fusion_method}\n\n"
            response += f"{result['summary']}\n\n"
            
            # Add extracted text if available
            if result['extracted_text']:
                response += f"**📝 Extracted Content:**\n{result['extracted_text']}\n\n"
            
            response += f"**📊 Analysis:**\n{result['analysis']}\n\n"
            response += f"🔗 [View Full Document](http://localhost:8000/document/{result['doc_id']})\n\n"
            response += "---\n\n"
        
        # Add remaining results as list
        if len(context.search_results) > 3:
            response += f"**📋 Additional Results:**\n"
            for i, result in enumerate(context.search_results[3:8], 4):
                doc_id = result.get("doc_id", "unknown")
                score = result.get("score", 0)
                sources = result.get("sources", [])
                fusion_method = result.get("fusion_method", "unknown")
                response += f"{i}. Document {doc_id} (confidence: {score:.2f})\n"
                response += f"   📊 Sources: {', '.join(sources)} | Fusion: {fusion_method}\n"
                response += f"   🔗 [View Document](http://localhost:8000/document/{doc_id})\n\n"
        
        if context.topics:
            response += f"🔍 **Related topics:** {', '.join([t.get('topic_name', '') for t in context.topics[:3]])}\n\n"
        
        if context.refined_queries:
            response += f"**🔧 Refined Queries:**\n"
            for i, query in enumerate(context.refined_queries[:3], 1):
                response += f"{i}. {query}\n"
            response += "\n"
        
        if context.suggestions:
            response += f"**💡 Search Suggestions:**\n"
            for i, suggestion in enumerate(context.suggestions[:3], 1):
                response += f"{i}. {suggestion}\n"
            response += "\n"
        
        response += "💡 **Tip:** I've analyzed the top results and extracted relevant content for you. Click any document link to view the full content!"
        
        return response
    
    async def _format_explanation_response(self, context: ConversationContext, original_query: str) -> str:
        """Format explanation response with document content analysis and extracted text"""
        response = "Based on analysis of the results:\n\n"
        
        if context.topics:
            response += f"Main topic: {context.topics[0].get('topic_name', 'unknown')}\n"
            response += f"Confidence level: {context.topics[0].get('relevance_score', 0):.2f}\n\n"
        
        if context.search_results:
            response += f"Number of matching results: {len(context.search_results)}\n"
            
            # Analyze top result in detail with extracted text
            if context.search_results:
                top_result = context.search_results[0]
                doc_id = top_result.get("doc_id", "unknown")
                doc_content = await self._read_document_content(doc_id, context.dataset)
                summary = await self._get_document_summary(doc_content)
                extracted_text = await self._extract_relevant_text(doc_content, original_query)
                
                response += f"\n**📄 Top Result Analysis:**\n"
                response += f"Document {doc_id}\n"
                response += f"{summary}\n\n"
                
                # Add extracted text if available
                if extracted_text and not extracted_text.startswith("❌"):
                    response += f"**📝 Extracted Content:**\n{extracted_text}\n\n"
                
                response += f"🔗 [View Full Document](http://localhost:8000/document/{doc_id})\n\n"
            
            response += "The best results are related to the requested topic.\n"
        
        return response
    
    async def _format_comparison_response(self, context: ConversationContext, original_query: str) -> str:
        """Format comparison response with document analysis and extracted text"""
        return "Comparison of results:\n\n" + await self._format_search_response(context, original_query)
    
    async def _format_analysis_response(self, context: ConversationContext, original_query: str) -> str:
        """Format analysis response with detailed document analysis and extracted text"""
        response = "Comprehensive analysis of results:\n\n"
        
        if context.topics:
            response += f"Discovered topics: {len(context.topics)}\n"
            for topic in context.topics[:3]:
                response += f"- {topic.get('topic_name', '')}: {topic.get('relevance_score', 0):.2f}\n"
        
        if context.search_results:
            response += f"\nNumber of results: {len(context.search_results)}\n"
            avg_score = sum(r.get('score', 0) for r in context.search_results) / len(context.search_results)
            response += f"Average confidence: {avg_score:.2f}\n\n"
            
            # Analyze top 2 results with extracted text
            for i, result in enumerate(context.search_results[:2], 1):
                doc_id = result.get("doc_id", "unknown")
                doc_content = await self._read_document_content(doc_id, context.dataset)
                analysis = await self._analyze_document_content(doc_content, original_query)
                extracted_text = await self._extract_relevant_text(doc_content, original_query)
                
                response += f"**📄 Result {i} Analysis:**\n"
                response += f"Document {doc_id}\n"
                response += f"{analysis}\n\n"
                
                # Add extracted text if available
                if extracted_text and not extracted_text.startswith("❌"):
                    response += f"**📝 Extracted Content:**\n{extracted_text}\n\n"
                
                response += f"🔗 [View Full Document](http://localhost:8000/document/{doc_id})\n\n"
        
        return response
    
    async def _format_refinement_response(self, context: ConversationContext, original_query: str) -> str:
        """Format refinement response"""
        response = "Search refinement:\n\n"
        
        if context.refined_queries:
            response += "Improved queries:\n"
            for i, query in enumerate(context.refined_queries[:3], 1):
                response += f"{i}. {query}\n"
        
        if context.suggestions:
            response += f"\nAdditional suggestions: {', '.join(context.suggestions[:3])}"
        
        return response
    
    async def _format_clarification_response(self, context: ConversationContext, original_query: str) -> str:
        """Format clarification response with document analysis and extracted text"""
        return "Additional clarification:\n\n" + await self._format_search_response(context, original_query)
    
    async def _format_summary_response(self, context: ConversationContext, original_query: str) -> str:
        """Format summary response with document summaries and extracted text"""
        response = "Summary of results:\n\n"
        
        if context.search_results:
            response += f"• Number of results: {len(context.search_results)}\n"
            
            # Summarize top result with extracted text
            top_result = context.search_results[0]
            doc_id = top_result.get("doc_id", "unknown")
            doc_content = await self._read_document_content(doc_id, context.dataset)
            summary = await self._get_document_summary(doc_content)
            extracted_text = await self._extract_relevant_text(doc_content, original_query)
            
            response += f"• Best result: Document {doc_id}\n"
            response += f"• {summary}\n"
            
            # Add extracted text if available
            if extracted_text and not extracted_text.startswith("❌"):
                response += f"• **Key Content:** {extracted_text[:200]}{'...' if len(extracted_text) > 200 else ''}\n"
            
            response += f"• 🔗 [View Full Document](http://localhost:8000/document/{doc_id})\n"
        
        if context.topics:
            response += f"• Main topics: {len(context.topics)}\n"
        
        return response
    
    async def _format_general_response(self, context: ConversationContext, original_query: str) -> str:
        """Format general response with document analysis and extracted text"""
        return await self._format_search_response(context, original_query)
    
    async def process_message(self, request: AgentRequest) -> AgentResponse:
        """Process user message with advanced AI reasoning and learning"""
        start_time = time.time()
        reasoning_steps = []
        
        try:
            # Generate session ID if not provided
            session_id = request.session_id or self._generate_session_id()
            
            # Get or create conversation context
            context = self._get_or_create_context(session_id, request.dataset)
            
            # Load user preferences from memory if user_id provided
            if request.user_id:
                context.user_id = request.user_id
                memory_data = self._load_from_memory(request.user_id)
                context.user_preferences.update(memory_data.get("preferences", {}))
                reasoning_steps.append("User preferences loaded from memory")
            
            # Detect language
            context.language_detected = self._detect_language(request.message)
            reasoning_steps.append(f"Language detected: {context.language_detected}")
            
            # Update context with advanced features
            context.current_state = ConversationState.UNDERSTANDING
            context.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "message": request.message,
                "user_id": request.user_id,
                "language": context.language_detected
            })
            
            reasoning_steps.append("Advanced conversation context created")
            
            # Advanced intent analysis with semantic understanding
            intent = self._analyze_intent(request.message)
            context.confidence_scores.append(intent["confidence"])
            reasoning_steps.append(f"Intent identified: {intent['primary_intent']} (confidence: {intent['confidence']:.2f})")
            if intent.get("semantic_analysis"):
                reasoning_steps.append("Semantic analysis performed")
            
            # Extract keywords with language-specific processing
            keywords = self._extract_keywords(request.message, context.language_detected)
            reasoning_steps.append(f"Keywords extracted: {', '.join(keywords[:5])}")
            
            # Generate refined queries with memory integration
            refined_queries = self._refine_query(request.message, context)
            context.refined_queries = refined_queries
            reasoning_steps.append(f"Generated {len(refined_queries)} refined queries")
            
            # Perform intelligent search
            context.current_state = ConversationState.SEARCHING
            search_results = []
            for query in refined_queries[:3]:  # Use top 3 refined queries
                results = await self._perform_search(query, request.dataset, request.max_results)
                search_results.extend(results)
            
            # Remove duplicates and limit results
            seen_ids = set()
            unique_results = []
            for result in search_results:
                result_id = result.get('id', result.get('title', ''))
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)
            context.search_results = unique_results[:request.max_results]
            reasoning_steps.append(f"Found {len(context.search_results)} unique results")
            
            # Detect topics if requested
            if request.include_topics:
                context.current_state = ConversationState.ANALYZING
                topics = await self._detect_topics(request.message, request.dataset)
                context.topics = topics
                reasoning_steps.append(f"Discovered {len(topics)} topics")
            
            # Get intelligent suggestions
            if request.include_suggestions:
                context.current_state = ConversationState.REFINING
                suggestions = await self._get_suggestions(request.message, request.dataset)
                context.suggestions = suggestions
                reasoning_steps.append(f"Generated {len(suggestions)} suggestions")
            
            # Synthesize intelligent response
            context.current_state = ConversationState.SYNTHESIZING
            response_text = await self._synthesize_response(context, intent, request.message)
            reasoning_steps.append("Advanced intelligent response synthesized")
            
            # Add agent response to conversation history
            context.conversation_history.append({
                "role": "agent",
                "message": response_text,
                "timestamp": datetime.now().isoformat(),
                "language": context.language_detected
            })
            
            # Learn and save to memory
            context.current_state = ConversationState.LEARNING
            if request.user_id:
                self._save_to_memory(request.user_id, context)
                reasoning_steps.append("Conversation saved to long-term memory")
            
            # Update final state
            context.current_state = ConversationState.COMPLETE
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                response=response_text,
                session_id=session_id,
                conversation_state=context.current_state.value,
                search_results=context.search_results,
                topics=context.topics,
                suggestions=context.suggestions,
                refined_queries=context.refined_queries,
                confidence_score=intent["confidence"],
                reasoning_steps=reasoning_steps,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return AgentResponse(
                response=f"Sorry, an error occurred while processing your message: {str(e)}",
                session_id=request.session_id or "error",
                conversation_state="error",
                confidence_score=0.0,
                reasoning_steps=["Error in processing"],
                execution_time=time.time() - start_time
            )

@app.post("/chat", response_model=AgentResponse)
async def chat_with_agent(request: AgentRequest):
    """Main chat endpoint for conversational agent"""
    agent = get_agent_instance()
    return await agent.process_message(request)

@app.post("/conversation", response_model=ConversationResponse)
async def continue_conversation(request: ConversationRequest):
    """Continue existing conversation"""
    agent = get_agent_instance()
    context = agent._get_or_create_context(request.session_id, request.dataset)
    
    # Process message
    agent_request = AgentRequest(
        message=request.message,
        session_id=request.session_id,
        dataset=request.dataset
    )
    
    response = await agent.process_message(agent_request)
    
    return ConversationResponse(
        session_id=request.session_id,
        response=response.response,
        context={
            "state": response.conversation_state,
            "topics": response.topics,
            "suggestions": response.suggestions,
            "refined_queries": response.refined_queries
        },
        suggestions=response.suggestions
    )

@app.get("/sessions")
async def get_active_sessions():
    """Get list of active conversation sessions"""
    agent = get_agent_instance()
    return {
        "active_sessions": len(agent.conversations),
        "sessions": [
            {
                "session_id": session_id,
                "dataset": context.dataset,
                "state": context.current_state.value,
                "history_length": len(context.conversation_history)
            }
            for session_id, context in agent.conversations.items()
        ]
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session"""
    agent = get_agent_instance()
    if session_id in agent.conversations:
        del agent.conversations[session_id]
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    agent = get_agent_instance()
    return {
        "status": "healthy",
        "service": "Professional AI Agent Service",
        "active_sessions": len(agent.conversations),
        "version": "2.0.0",
        "features": {
            "semantic_analysis": agent.embedding_model is not None,
            "memory_database": agent.memory_db is not None,
            "language_detection": True,
            "long_term_memory": True,
            "multi_language_support": True
        }
    }

# WebSocket support for real-time conversation
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message through agent
            agent = get_agent_instance()
            request = AgentRequest(
                message=message_data.get("message", ""),
                session_id=session_id,
                dataset=message_data.get("dataset", "argsme")
            )
            
            response = await agent.process_message(request)
            
            # Send response back
            await manager.send_personal_message(
                json.dumps({
                    "response": response.response,
                    "session_id": response.session_id,
                    "state": response.conversation_state,
                    "suggestions": response.suggestions,
                    "topics": response.topics
                }),
                websocket
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011) 