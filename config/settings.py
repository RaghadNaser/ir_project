# Application settings and configuration

# Dataset configurations
DATASETS = {
    "argsme": {
        "name": "argsme",
        "display_name": "ARGSME Dataset",
        "description": "Argument mining dataset"
    },
    "wikir": {
        "name": "wikir", 
        "display_name": "WikiR Dataset",
        "description": "Wikipedia retrieval dataset"
    },
    "wikir_en1k_training": {
        "name": "wikir_en1k_training",
        "display_name": "WikiR EN1K Training",
        "description": "WikiR English 1K training dataset"
    }
}

# BERT model configuration
BERT_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "max_length": 512,
    "batch_size": 32
}

# TF-IDF configuration
TFIDF_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95
}

# Hybrid search configuration
HYBRID_CONFIG = {
    "default_tfidf_weight": 0.4,
    "default_embedding_weight": 0.6,
    "default_fusion_candidates_k": 1000,
    "default_first_stage_k": 2000
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "reload": True
} 