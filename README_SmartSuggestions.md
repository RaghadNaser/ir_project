# Smart Query Suggestion Service

## Overview

The Smart Query Suggestion Service is an advanced system for providing intelligent and diverse query suggestions using multiple advanced methods. It was designed to solve the problem of limited original queries (only 50 queries in ARGSME) by providing smart and diverse suggestions.

## Features

### 1. Smart Suggestions
- Combines multiple methods in one suggestion
- Smart distribution: 30% similar, 25% related, 20% auto-complete, 15% correction, 10% category

### 2. Popular Queries
- Calculate popularity scores based on:
  - Query length (medium length queries are more popular)
  - Presence of important keywords
  - Query type (original queries are more popular)

### 3. Similar Queries
- Uses TF-IDF and Cosine Similarity
- Search for semantically similar queries
- Similarity threshold of 0.1 for quality

### 4. Trending Queries
- Analyze keyword frequency patterns
- Create trending queries based on patterns
- Example: "What is climate change?"

### 5. Related Queries
- Calculate keyword overlap
- Search for related topics
- Relatedness threshold of 0.2

### 6. Auto-complete
- Search for prefix match
- Search for contains match
- Sort by match type

### 7. Query Expansion
- Use WordNet for synonyms
- Expand queries with synonymous words
- Example: "education" -> "learning"

### 8. Spell Correction
- Use difflib to search for similar queries
- Correct spelling errors
- Similarity threshold of 0.6

### 9. Category-based Suggestions
- Classify queries into categories:
  - Education
  - Health
  - Technology
  - Politics
  - Environment
  - Economy

## Installation and Usage

### 1. Start Smart Service
```bash
python services/query_suggestion_service/start.py
```

### 2. Test Service
```bash
python test_smart_suggestions.py
```

### 3. Access Service
- **Service**: http://localhost:8010
- **Documentation**: http://localhost:8010/docs
- **Statistics**: http://localhost:8010/stats
- **Available Methods**: http://localhost:8010/methods

## API Endpoints

### POST /suggest
```json
{
    "query": "Should teachers get tenure?",
    "dataset": "argsme",
    "method": "smart",
    "top_k": 10,
    "include_metadata": true
}
```

### GET /stats
```json
{
    "argsme": {
        "total_queries": 300,
        "original_queries": 50,
        "enhanced_queries": 250,
        "popularity_scores_calculated": true,
        "similarity_matrix_built": true
    },
    "wikir": {
        "total_queries": 1446,
        "original_queries": 1446,
        "enhanced_queries": 0,
        "popularity_scores_calculated": true,
        "similarity_matrix_built": true
    }
}
```

### GET /methods
```json
{
    "methods": [
        {
            "name": "smart",
            "description": "Smart suggestions combining multiple methods",
            "best_for": "General use"
        },
        {
            "name": "popular",
            "description": "Most popular queries",
            "best_for": "Discovering common topics"
        }
    ]
}
```

## Usage Examples

### 1. Smart Suggestions
```python
import requests

response = requests.post("http://localhost:8010/suggest", json={
    "query": "climate change",
    "dataset": "argsme",
    "method": "smart",
    "top_k": 8
})

suggestions = response.json()["suggestions"]
for suggestion in suggestions:
    print(f"- {suggestion['query']} (Score: {suggestion['score']:.2f}, Type: {suggestion['type']})")
```

### 2. Similar Suggestions
```python
response = requests.post("http://localhost:8010/suggest", json={
    "query": "Should teachers get tenure?",
    "dataset": "argsme",
    "method": "similar",
    "top_k": 5
})
```

### 3. Popular Suggestions
```python
response = requests.post("http://localhost:8010/suggest", json={
    "query": "",
    "dataset": "argsme",
    "method": "popular",
    "top_k": 10
})
```

## Applied Improvements

### 1. Query Expansion
- From 50 queries to 300+ queries
- Create additional queries from existing titles
- Various patterns: "Should X?", "What is X?", "How to X?"

### 2. Popularity Calculation
- Advanced algorithm for popularity calculation
- Multiple factors: length, keywords, type
- Smart ranking of suggestions

### 3. Similarity Matrix
- Build similarity matrix once
- Cache for optimization
- TF-IDF with n-grams

### 4. Duplicate Removal
- Algorithm to remove duplicate queries
- Sort by score
- Diversify suggestions

## Customization

### Add New Methods
```python
def get_custom_suggestions(self, query: str, dataset: str, top_k: int) -> List[Dict]:
    """Custom new method"""
    # Implement custom logic
    return suggestions
```

### Modify Smart Suggestion Weights
```python
def get_smart_suggestions(self, query: str, dataset: str, top_k: int) -> List[Dict]:
    # Modify ratios as needed
    similar_weight = 0.3      # 30%
    related_weight = 0.25     # 25%
    autocomplete_weight = 0.2 # 20%
    correction_weight = 0.15  # 15%
    category_weight = 0.1     # 10%
```

## Statistics

### Before Improvement
- **ARGSME**: Only 50 queries
- **WIKIR**: 1446 queries
- **Suggestion methods**: Limited

### After Improvement
- **ARGSME**: 300+ queries
- **WIKIR**: 1446 queries
- **Suggestion methods**: 9 advanced methods
- **Suggestion quality**: Significantly improved
- **Suggestion diversity**: High

## Usage in Interface

The user interface has been updated to include all new methods:

```html
<select id="suggestion_method" onchange="fetchSuggestions()">
    <option value="smart">Smart (All Methods)</option>
    <option value="popular">Popular</option>
    <option value="similar">Similar</option>
    <option value="trending">Trending</option>
    <option value="related">Related</option>
    <option value="autocomplete">Auto-complete</option>
    <option value="expansion">Expansion</option>
    <option value="correction">Correction</option>
    <option value="category">Category</option>
</select>
```

## Expected Results

1. **Greater diversity** in suggestions
2. **Better quality** of suggestions
3. **Improved user experience**
4. **Broader topic coverage**
5. **Flexibility in customization**

---

**This service was developed to solve the problem of limited original queries and provide smart and diverse suggestions for users.** 