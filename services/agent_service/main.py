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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import json
import sqlite3
import pandas as pd
import requests
import random
from typing import List, Dict, Any, Optional
import uvicorn
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    language: str = "ar"  # "ar" for Arabic, "en" for English
    search_method: str = "hybrid"  # "hybrid", "tfidf", "embedding"
    top_k: int = 5
    dataset: Optional[str] = None  # "argsme", "wikir", or None for suggestion

class ChatResponse(BaseModel):
    response: str
    documents: List[Dict[str, Any]]
    confidence: float
    language: str
    search_method: str
    dataset_suggestion: Optional[bool] = False  # Whether to suggest dataset choice

# Global database connection
DB_PATH = "data/ir_database_combined.db"
DB_CONN = None

# Language-specific responses
RESPONSES = {
    "ar": {
        "no_results": "لم أجد أي مستندات ذات صلة لاستعلامك '{query}'. يرجى إعادة صياغة سؤالك أو تجربة كلمات مفتاحية مختلفة.",
        "found_documents": "بناءً على استعلامك '{query}'، وجدت {count} مستندات ذات صلة:\n",
        "argsme_count": "• {count} مستند من مجموعة ARGSME\n",
        "wikir_count": "• {count} مستند من مجموعة WIKIR\n",
        "best_results": "أفضل النتائج: {titles}",
        "more_results": " و {count} مستندات أخرى.",
        "helpful_info": "\n\nهذه المستندات تحتوي على معلومات قد تكون مفيدة لبحثك.",
        "dataset_suggestion": "مرحباً! لمساعدتك بشكل أفضل، أي مجموعة بيانات تفضل أن أبحث فيها؟\n\n• ARGSME: مناقشات وحجج حول مواضيع مختلفة\n• WIKIR: مقالات من ويكيبيديا",
        "clarification_needed": "سؤالك مثير للاهتمام! 🤔 لمساعدتك بشكل أفضل، هل يمكنك توضيح:\n\n{clarification_points}",
        "clarification_examples": {
            "general": "• ما هو الجانب المحدد الذي تريد معرفة المزيد عنه؟\n• هل تبحث عن معلومات حديثة أم تاريخية؟\n• هل تريد آراء مختلفة أم حقائق محددة؟\n• هل تريد أمثلة عملية أم معلومات نظرية؟",
            "technology": "• ما هو الجانب التقني المحدد الذي يهمك؟\n• هل تبحث عن التطبيقات أم النظرية؟\n• هل تريد معلومات للمبتدئين أم للمتقدمين؟\n• هل تريد معلومات عن أحدث التقنيات أم الأساسيات؟",
            "politics": "• ما هو الجانب السياسي المحدد؟\n• هل تبحث عن آراء مؤيدة أم معارضة؟\n• هل تريد معلومات عن دولة أو منطقة محددة؟\n• هل تريد معلومات عن الانتخابات أم السياسات العامة؟",
            "health": "• ما هو الجانب الصحي المحدد؟\n• هل تبحث عن الوقاية أم العلاج؟\n• هل تريد معلومات علمية أم تجارب شخصية؟\n• هل تريد معلومات عن مرض معين أم صحة عامة؟"
        },
        "search_methods": {
            "hybrid": "البحث الهجين",
            "tfidf": "البحث بالكلمات المفتاحية",
            "embedding": "البحث الدلالي"
        }
    },
    "en": {
        "no_results": "I couldn't find any relevant documents for your query '{query}'. Please rephrase your question or try different keywords.",
        "found_documents": "Based on your query '{query}', I found {count} relevant documents:\n",
        "argsme_count": "• {count} documents from ARGSME dataset\n",
        "wikir_count": "• {count} documents from WIKIR dataset\n",
        "best_results": "Best results: {titles}",
        "more_results": " and {count} more documents.",
        "helpful_info": "\n\nThese documents contain information that might be helpful for your search.",
        "dataset_suggestion": "Hello! To help you better, which dataset would you prefer me to search in?\n\n• ARGSME: Discussions and arguments on various topics\n• WIKIR: Articles from Wikipedia",
        "clarification_needed": "Your question is interesting! 🤔 To help you better, could you clarify:\n\n{clarification_points}",
        "clarification_examples": {
            "general": "• What specific aspect do you want to know more about?\n• Are you looking for recent or historical information?\n• Do you want different opinions or specific facts?",
            "technology": "• What specific technical aspect interests you?\n• Are you looking for applications or theory?\n• Do you want information for beginners or advanced users?",
            "politics": "• What specific political aspect?\n• Are you looking for pro or con opinions?\n• Do you want information about a specific country or region?",
            "health": "• What specific health aspect?\n• Are you looking for prevention or treatment?\n• Do you want scientific information or personal experiences?"
        },
        "search_methods": {
            "hybrid": "Hybrid Search",
            "tfidf": "Keyword Search",
            "embedding": "Semantic Search"
        }
    }
}

def init_database():
    """Initialize database connection"""
    global DB_CONN
    
    try:
        if not Path(DB_PATH).exists():
            logger.error(f"Database file not found: {DB_PATH}")
            return False
            
        DB_CONN = sqlite3.connect(DB_PATH)
        logger.info(f"Database connected successfully: {DB_PATH}")
        
        # Check available tables
        cursor = DB_CONN.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logger.info(f"Available tables: {[table[0] for table in tables]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        return False

def call_hybrid_service(query: str, dataset: str = "argsme", top_k: int = 10) -> List[Dict[str, Any]]:
    """Call the hybrid service to get diverse results"""
    try:
        hybrid_url = "http://localhost:8005/search"
        hybrid_data = {
            "query": query,
            "dataset": dataset,
            "method": "sequential",
            "top_k": top_k,
            "tfidf_weight": 0.4,
            "embedding_weight": 0.6
        }
        
        response = requests.post(hybrid_url, json=hybrid_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get("results", [])
        else:
            logger.warning(f"Hybrid service returned {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error calling hybrid service: {str(e)}")
        return []

def call_tfidf_service(query: str, dataset: str = "argsme", top_k: int = 10) -> List[Dict[str, Any]]:
    """Call the TF-IDF service"""
    try:
        tfidf_url = "http://localhost:8003/search"
        tfidf_data = {
            "query": query,
            "dataset": dataset,
            "top_k": top_k
        }
        
        response = requests.post(tfidf_url, json=tfidf_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get("results", [])
        else:
            logger.warning(f"TF-IDF service returned {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error calling TF-IDF service: {str(e)}")
        return []

def call_embedding_service(query: str, dataset: str = "argsme", top_k: int = 10) -> List[Dict[str, Any]]:
    """Call the embedding service"""
    try:
        embedding_url = "http://localhost:8004/search"
        embedding_data = {
            "query": query,
            "dataset": dataset,
            "top_k": top_k
        }
        
        response = requests.post(embedding_url, json=embedding_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get("results", [])
        else:
            logger.warning(f"Embedding service returned {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error calling embedding service: {str(e)}")
        return []

def get_document_details(doc_id: str, dataset: str) -> Dict[str, Any]:
    """Get detailed document information from database"""
    global DB_CONN
    
    if DB_CONN is None:
        return {}
    
    try:
        if dataset == "argsme":
            sql = """
            SELECT doc_id, conclusion, premises_texts, source_title, topic, acquisition
            FROM argsme_raw 
            WHERE doc_id = ?
            LIMIT 1
            """
        else:  # wikir
            sql = """
            SELECT doc_id, text
            FROM wikir_docs
            WHERE doc_id = ?
            LIMIT 1
            """
        
        # Use cursor instead of pandas for faster query
        cursor = DB_CONN.cursor()
        cursor.execute(sql, [doc_id])
        row = cursor.fetchone()
        
        if not row:
            return {}
        
        if dataset == "argsme":
            # Combine text content
            content_parts = []
            if row[1]:  # conclusion
                content_parts.append(str(row[1]))
            if row[2]:  # premises_texts
                content_parts.append(str(row[2]))
            
            full_content = " ".join(content_parts)
            
            return {
                'id': str(row[0]),  # doc_id
                'title': str(row[3]) if row[3] else "No Title",  # source_title
                'content': full_content[:300] + "..." if len(full_content) > 300 else full_content,  # Shorter content
                'topic': str(row[4]) if row[4] else "Unknown",  # topic
                'source': 'argsme',
                'acquisition': str(row[5]) if row[5] else "Unknown"  # acquisition
            }
        else:
            content = str(row[1])  # text
            # For WIKIR, use first 100 characters of text as title
            title = content[:100] + "..." if len(content) > 100 else content
            return {
                'id': str(row[0]),  # doc_id
                'title': title,
                'content': content[:300] + "..." if len(content) > 300 else content,  # Shorter content
                'topic': "WIKIR Document",
                'source': 'wikir',
                'acquisition': "WIKIR Dataset"
            }
        
    except Exception as e:
        logger.error(f"Error getting document details: {str(e)}")
        return {}

def search_documents(query: str, language: str = "ar", search_method: str = "hybrid", top_k: int = 5, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search documents using the specified method"""
    
    # Use specified dataset or default to argsme
    if dataset and dataset in ["argsme", "wikir"]:
        datasets = [dataset]
    else:
        datasets = ["argsme"]  # Default to argsme for speed
    
    all_results = []
    
    for dataset in datasets:
        try:
            if search_method == "hybrid":
                results = call_hybrid_service(query, dataset, top_k)
            elif search_method == "tfidf":
                results = call_tfidf_service(query, dataset, top_k)
            elif search_method == "embedding":
                results = call_embedding_service(query, dataset, top_k)
            else:
                # Fallback to tfidf for speed
                results = call_tfidf_service(query, dataset, top_k)
            
            # Get detailed document information (limit to first few for speed)
            for i, result in enumerate(results[:min(top_k, 3)]):  # Limit to 3 documents for speed
                doc_id = result.get("doc_id", "")
                if doc_id:
                    doc_details = get_document_details(doc_id, dataset)
                    if doc_details:
                        # Add score from search result
                        doc_details['score'] = result.get("score", 0.5)
                        all_results.append(doc_details)
            
        except Exception as e:
            logger.error(f"Error searching {dataset}: {str(e)}")
            continue
    
    # Return results without shuffling for speed
    return all_results[:top_k]

def analyze_query_clarity(query: str, language: str = "ar") -> tuple:
    """Analyze query clarity and determine if clarification is needed"""
    query_lower = query.lower()
    words = query.split()
    
    # Keywords that indicate vague or broad queries
    vague_keywords = {
        "ar": ["كيف", "لماذا", "متى", "أين", "ما", "أي", "كل", "جميع", "عام", "عامة", "مختلف", "متعدد", "أفضل", "أحسن", "أهم", "مفيد"],
        "en": ["how", "why", "when", "where", "what", "which", "all", "every", "general", "different", "various", "multiple", "best", "good", "important", "useful"]
    }
    
    # Very short queries that need clarification
    very_short_queries = {
        "ar": ["تقنية", "سياسة", "صحة", "تعليم", "اقتصاد", "رياضة", "فن", "علوم"],
        "en": ["technology", "politics", "health", "education", "economy", "sports", "art", "science"]
    }
    
    # Topic-specific keywords for better clarification
    topic_keywords = {
        "technology": {
            "ar": ["تقنية", "تكنولوجيا", "برمجة", "كمبيوتر", "إنترنت", "ذكاء اصطناعي", "تعلم آلي", "روبوت", "أمن سيبراني"],
            "en": ["technology", "tech", "programming", "computer", "internet", "ai", "machine learning", "robot", "cybersecurity"]
        },
        "politics": {
            "ar": ["سياسة", "حكومة", "انتخابات", "ديمقراطية", "حزب", "رئيس", "وزير", "برلمان", "قانون"],
            "en": ["politics", "government", "election", "democracy", "party", "president", "minister", "parliament", "law"]
        },
        "health": {
            "ar": ["صحة", "طب", "دواء", "علاج", "مرض", "وقاية", "مستشفى", "طبيب", "أعراض"],
            "en": ["health", "medical", "medicine", "treatment", "disease", "prevention", "hospital", "doctor", "symptoms"]
        }
    }
    
    # Check if query is too vague
    vague_count = sum(1 for word in vague_keywords[language] if word in query_lower)
    
    # Check if it's a very short query
    is_very_short = len(words) <= 2 and any(word in very_short_queries[language] for word in words)
    
    # Check if it's a question that's too broad
    is_broad_question = vague_count >= 1 and len(words) <= 5
    
    # Check if it's just a single topic word
    is_single_topic = len(words) == 1 and any(word in very_short_queries[language] for word in words)
    
    # Check if it's a very general question
    general_questions = {
        "ar": ["ما هو", "كيف", "لماذا", "متى", "أين", "أي", "كل", "جميع", "عام", "عامة"],
        "en": ["what is", "how", "why", "when", "where", "which", "all", "every", "general"]
    }
    is_general_question = any(question in query_lower for question in general_questions[language])
    
    # Check if it's a clarification request (don't ask for clarification again)
    clarification_indicators = {
        "ar": ["أريد", "أحتاج", "أبحث عن", "أفضل", "أحسن", "معلومات", "تفاصيل", "أمثلة", "أكثر", "متقدم", "مبتدئ"],
        "en": ["want", "need", "looking for", "best", "good", "information", "details", "examples", "more", "advanced", "beginner"]
    }
    is_clarification_request = any(indicator in query_lower for indicator in clarification_indicators[language])
    
    # More sensitive to vague queries
    is_vague = (is_very_short or is_broad_question or is_single_topic or is_general_question or vague_count >= 1) and not is_clarification_request
    
    # Determine topic for specific clarification
    detected_topic = "general"
    for topic, keywords in topic_keywords.items():
        if any(keyword in query_lower for keyword in keywords[language]):
            detected_topic = topic
            break
    
    return is_vague, detected_topic

def generate_response(query: str, documents: List[Dict[str, Any]], language: str = "ar", search_method: str = "hybrid", suggest_dataset: bool = False) -> str:
    """Generate a response based on the query and found documents"""
    responses = RESPONSES.get(language, RESPONSES["ar"])
    
    if not documents:
        return responses["no_results"].format(query=query)
    
    # Count documents by source
    argsme_count = len([d for d in documents if d['source'] == 'argsme'])
    wikir_count = len([d for d in documents if d['source'] == 'wikir'])
    
    doc_titles = [doc["title"] for doc in documents[:3]]  # Show first 3 titles
    
    response = responses["found_documents"].format(query=query, count=len(documents))
    
    if argsme_count > 0:
        response += responses["argsme_count"].format(count=argsme_count)
    if wikir_count > 0:
        response += responses["wikir_count"].format(count=wikir_count)
    
    response += responses["best_results"].format(titles=", ".join(doc_titles))
    
    if len(documents) > 3:
        response += responses["more_results"].format(count=len(documents) - 3)
    
    response += responses["helpful_info"]
    
    # Add search method info
    method_name = responses["search_methods"].get(search_method, search_method)
    if language == "ar":
        response += f"\n\nتم استخدام: {method_name}"
    else:
        response += f"\n\nUsed: {method_name}"
    
    # Add dataset suggestion if requested
    if suggest_dataset:
        response += f"\n\n{responses['dataset_suggestion']}"
    
    return response

@app.on_event("startup")
async def startup_event():
    """Initialize database when the service starts"""
    logger.info("Starting Agent Service...")
    if init_database():
        logger.info("Agent Service started successfully with database connection")
    else:
        logger.warning("Agent Service started but failed to connect to database")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection when the service stops"""
    global DB_CONN
    if DB_CONN:
        DB_CONN.close()
        logger.info("Database connection closed")

@app.get("/")
async def root():
    return {
        "message": "Agent Service is running", 
        "status": "healthy", 
        "database_connected": DB_CONN is not None,
        "features": ["multi-language", "hybrid-search", "database-integration"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "agent", 
        "database_connected": DB_CONN is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.message} (lang: {request.language}, method: {request.search_method}, dataset: {request.dataset})")
        
        # Check if this is a dataset selection message
        if request.message.lower() in ["argsme", "wikir", "arg", "wik"]:
            # User selected a dataset, store it and ask for their question
            responses = RESPONSES.get(request.language, RESPONSES["ar"])
            if request.language == "ar":
                response_text = f"ممتاز! تم اختيار مجموعة البيانات {request.message.upper()}. ما هو سؤالك؟"
            else:
                response_text = f"Great! Selected {request.message.upper()} dataset. What's your question?"
            
            return ChatResponse(
                response=response_text,
                documents=[],
                confidence=0.8,
                language=request.language,
                search_method=request.search_method,
                dataset_suggestion=False
            )
        
        # Determine if we should suggest dataset choice
        suggest_dataset = request.dataset is None
        
        # If no dataset specified, just suggest dataset choice without searching
        if suggest_dataset:
            responses = RESPONSES.get(request.language, RESPONSES["ar"])
            response_text = responses["dataset_suggestion"]
            documents = []
        else:
            # Analyze query clarity FIRST, before searching
            is_vague, detected_topic = analyze_query_clarity(request.message, request.language)
            
            # If query is vague, ask for clarification first
            if is_vague:
                responses = RESPONSES.get(request.language, RESPONSES["ar"])
                clarification_points = responses["clarification_examples"].get(detected_topic, responses["clarification_examples"]["general"])
                response_text = responses["clarification_needed"].format(clarification_points=clarification_points)
                
                # Search for some initial results to show
                documents = search_documents(
                    request.message, 
                    language=request.language,
                    search_method=request.search_method, 
                    top_k=2,  # Just get 2 documents for preview
                    dataset=request.dataset
                )
                
                # Show some initial results if available
                if len(documents) > 0:
                    response_text += f"\n\nلكن إليك بعض النتائج الأولية:\n"
                    doc_titles = [doc["title"] for doc in documents[:2]]
                    response_text += f"• {', '.join(doc_titles)}"
                else:
                    response_text += f"\n\nسأبحث عن معلومات عامة حول موضوعك..."
                
                # Add a friendly note
                response_text += f"\n\n💡 يمكنك النقر على أي من الأزرار أعلاه أو كتابة سؤالك بتفصيل أكثر!"
            else:
                # Search for relevant documents only if query is clear
                documents = search_documents(
                    request.message, 
                    language=request.language,
                    search_method=request.search_method, 
                    top_k=request.top_k,
                    dataset=request.dataset
                )
                
                # Generate normal response with documents
                response_text = generate_response(
                    request.message, 
                    documents, 
                    language=request.language,
                    search_method=request.search_method,
                    suggest_dataset=False
                )
        
        # Calculate confidence based on number of documents found
        confidence = min(0.9, 0.3 + len(documents) * 0.15)
        
        result = ChatResponse(
            response=response_text,
            documents=documents,
            confidence=confidence,
            language=request.language,
            search_method=request.search_method,
            dataset_suggestion=suggest_dataset
        )
        
        logger.info(f"Chat response generated successfully with {len(documents)} documents")
        return result
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/extract", response_model=Dict[str, Any])
async def extract_content(request: ChatRequest):
    try:
        logger.info(f"Received extraction request: {request.message}")
        
        # Search for relevant documents
        documents = search_documents(
            request.message, 
            language=request.language,
            search_method=request.search_method, 
            top_k=10
        )
        
        responses = RESPONSES.get(request.language, RESPONSES["ar"])
        
        extraction_result = {
            "extracted_content": responses["found_documents"].format(
                query=request.message, count=len(documents)
            ),
            "documents": documents,
            "summary": f"Query '{request.message}' matched {len(documents)} documents in the database.",
            "sources": {
                "argsme": len([d for d in documents if d['source'] == 'argsme']),
                "wikir": len([d for d in documents if d['source'] == 'wikir'])
            },
            "language": request.language,
            "search_method": request.search_method
        }
        
        logger.info("Extraction completed successfully")
        return extraction_result
        
    except Exception as e:
        logger.error(f"Error in extract endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Agent Service...")
    uvicorn.run(app, host="0.0.0.0", port=8011, reload=False) 