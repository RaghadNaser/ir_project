# Configuration settings for the IR system

# Performance settings
ENABLE_RESULT_ENHANCEMENT = False  # Set to True to enable full document enhancement (slower)
ENABLE_TOPIC_DETECTION = False     # Set to True to enable topic detection (slower)
ENABLE_QUERY_SUGGESTIONS = False   # Set to True to enable query suggestions (slower)

# Database settings
DB_PATH = "data/ir_database_combined.db"

# Service URLs
SERVICE_URLS = {
    "preprocessing": "http://localhost:8002",
    "tfidf": "http://localhost:8003",
    "embedding": "http://localhost:8004",
    "hybrid": "http://localhost:8005",
    "topic_detection": "http://localhost:8006",
    "query_suggestions": "http://localhost:8010",
    "agent": "http://localhost:8011"
}

# Search service map
SEARCH_SERVICE_URLS = {
    "tfidf": "http://localhost:8003/search",
    "embedding": "http://localhost:8004/search",
    "hybrid": "http://localhost:8005/search"
}

# Dataset list
DATASETS = ["argsme", "wikir"]

# Representations
REPRESENTATIONS = [
    ("tfidf", "TF-IDF"),
    ("embedding", "Embedding"),
    ("hybrid", "Hybrid (Enhanced)")
]

# Hybrid search methods
HYBRID_METHODS = [
    ("sequential", "Sequential Search"),
    ("parallel", "Parallel Search"),
    ("fusion", "Fusion Search")
]

# Performance modes
PERFORMANCE_MODES = {
    "fast": {
        "description": "Fast mode - basic information only",
        "features": ["Basic titles", "No database queries", "No topic detection"],
        "speed": "Very fast"
    },
    "balanced": {
        "description": "Balanced mode - moderate features",
        "features": ["Enhanced titles", "Basic topic detection", "Query suggestions"],
        "speed": "Moderate"
    },
    "full": {
        "description": "Full mode - all features enabled",
        "features": ["Full document enhancement", "Advanced topic detection", "All suggestions"],
        "speed": "Slower"
    }
} 