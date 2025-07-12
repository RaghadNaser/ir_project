"""
Agent Service Utilities
Helper functions for the conversational agent
"""

import re
import json
from typing import List, Dict, Any
from datetime import datetime

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep Arabic and English
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    return text

def extract_entities(text: str) -> List[str]:
    """Extract potential entities from text"""
    # Simple entity extraction (can be enhanced with NER)
    words = text.split()
    entities = []
    
    for word in words:
        # Check for capitalized words (potential entities)
        if word[0].isupper() and len(word) > 2:
            entities.append(word)
        # Check for words with numbers
        elif any(char.isdigit() for char in word):
            entities.append(word)
    
    return entities

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def create_session_summary(context: Dict[str, Any]) -> str:
    """Create a summary of the conversation session"""
    summary = "Session Summary:\n\n"
    
    if context.get("search_results"):
        summary += f"• Number of results: {len(context['search_results'])}\n"
    
    if context.get("topics"):
        summary += f"• Discovered topics: {len(context['topics'])}\n"
    
    if context.get("suggestions"):
        summary += f"• Suggestions: {len(context['suggestions'])}\n"
    
    return summary

def validate_query(query: str) -> bool:
    """Validate if query is meaningful"""
    if not query or len(query.strip()) < 2:
        return False
    
    # Check if query has meaningful content
    words = query.split()
    if len(words) < 1:
        return False
    
    return True

def calculate_confidence_score(results: List[Dict], topics: List[Dict]) -> float:
    """Calculate confidence score based on results and topics"""
    if not results:
        return 0.0
    
    # Base confidence from search results
    avg_score = sum(r.get('score', 0) for r in results) / len(results)
    
    # Boost confidence if topics are detected
    topic_boost = min(len(topics) * 0.1, 0.3)
    
    return min(avg_score + topic_boost, 1.0) 