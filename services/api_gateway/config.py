"""
Configuration file for API Gateway
Contains service endpoints and settings for the enhanced professional interface
"""

import os
from typing import Dict, Any

class ServiceConfig:
    """Service configuration for the IR System"""
    
    # Service URLs
    SERVICES = {
        'main': 'http://localhost:8000',
        'tfidf': 'http://localhost:8003',
        'hybrid': 'http://localhost:8005',
        'indexing': 'http://localhost:8001',
        'embedding': 'http://localhost:8004',
        'preprocessing': 'http://localhost:8002',
        'unified_search': 'http://localhost:8009',
        'topic_detection': 'http://localhost:8006',
        'vector_store': 'http://localhost:8008',
        'embedding_vector_store': 'http://localhost:8008',
        'query_suggestion': 'http://localhost:8010'
    }
    
    # Service ports
    PORTS = {
        'api_gateway': 8000,
        'tfidf_service': 8003,
        'hybrid_service': 8005,
        'indexing_service': 8001,
        'embedding_service': 8004,
        'preprocessing_service': 8002,
        'unified_search_service': 8009,
        'topic_detection_service': 8006,
        'vector_store_service': 8008,
        'embedding_vector_store_service': 8008,
        'query_suggestion_service': 8010
    }
    
    # Service descriptions
    SERVICE_DESCRIPTIONS = {
        'main': 'Main API Gateway - Coordinates all search operations',
        'tfidf': 'TF-IDF Service - Traditional text search using TF-IDF vectors',
        'hybrid': 'Hybrid Service - Combines embedding and TF-IDF approaches',
        'indexing': 'Indexing Service - Manages document indexing and retrieval',
        'embedding': 'Embedding Service - Generates semantic embeddings',
        'preprocessing': 'Preprocessing Service - Cleans and processes text data',
        'unified_search': 'Unified Search - Advanced search with multiple backends',
        'topic_detection': 'Topic Detection - Identifies topics in queries and documents',
        'vector_store': 'Vector Store - High-performance FAISS-based similarity search',
        'query_suggestion': 'Query Suggestion - Smart query recommendations'
    }
    
    # Service features
    SERVICE_FEATURES = {
        'topic_detection': {
            'name': 'Topic Detection',
            'icon': 'fas fa-brain',
            'description': 'Automatically detect and analyze topics in your search results using advanced NLP models.',
            'config_options': {
                'max_topics': {
                    'type': 'number',
                    'default': 5,
                    'min': 1,
                    'max': 20,
                    'label': 'Max Topics'
                },
                'min_score': {
                    'type': 'number',
                    'default': 0.1,
                    'min': 0,
                    'max': 1,
                    'step': 0.1,
                    'label': 'Min Score'
                }
            }
        },
        'query_suggestion': {
            'name': 'Smart Suggestions',
            'icon': 'fas fa-lightbulb',
            'description': 'Get intelligent query suggestions based on semantic similarity and user patterns.',
            'config_options': {
                'method': {
                    'type': 'select',
                    'default': 'hybrid',
                    'options': [
                        {'value': 'hybrid', 'label': 'Hybrid (Best)'},
                        {'value': 'semantic', 'label': 'Semantic'},
                        {'value': 'popular', 'label': 'Popular'},
                        {'value': 'autocomplete', 'label': 'Autocomplete'}
                    ],
                    'label': 'Method'
                },
                'count': {
                    'type': 'number',
                    'default': 8,
                    'min': 1,
                    'max': 20,
                    'label': 'Count'
                }
            }
        },
        'vector_store': {
            'name': 'Vector Store',
            'icon': 'fas fa-rocket',
            'description': 'Use high-performance FAISS vector store for ultra-fast semantic search.',
            'config_options': {
                'index_type': {
                    'type': 'select',
                    'default': 'auto',
                    'options': [
                        {'value': 'auto', 'label': 'Auto (Best)'},
                        {'value': 'hnsw', 'label': 'HNSW (Fast)'},
                        {'value': 'ivf', 'label': 'IVF (Balanced)'},
                        {'value': 'flat', 'label': 'Flat (Accurate)'}
                    ],
                    'label': 'Index'
                },
                'performance': {
                    'type': 'select',
                    'default': 'balanced',
                    'options': [
                        {'value': 'balanced', 'label': 'Balanced'},
                        {'value': 'speed', 'label': 'Speed'},
                        {'value': 'accuracy', 'label': 'Accuracy'}
                    ],
                    'label': 'Performance'
                }
            }
        }
    }
    
    # Default search parameters
    DEFAULT_SEARCH_PARAMS = {
        'dataset': 'argsme',
        'top_k': 10,
        'representation': 'hybrid',
        'timeout': 30
    }
    
    # Supported datasets
    DATASETS = {
        'argsme': {
            'name': 'ARGSME',
            'description': 'Argumentative dataset for debate and argument analysis',
            'full_name': 'ARGSME (Argumentative)'
        },
        'wikir': {
            'name': 'WIKIR',
            'description': 'Wikipedia-based information retrieval dataset',
            'full_name': 'WIKIR (Wikipedia)'
        }
    }
    
    # Search representations
    REPRESENTATIONS = {
        'hybrid': {
            'name': 'Hybrid',
            'description': 'Combines embedding and TF-IDF approaches for best results',
            'full_name': 'Hybrid (Recommended)'
        },
        'embedding': {
            'name': 'Embedding',
            'description': 'Uses semantic embeddings for meaning-based search',
            'full_name': 'Embedding'
        },
        'tfidf': {
            'name': 'TF-IDF',
            'description': 'Traditional keyword-based search using TF-IDF',
            'full_name': 'TF-IDF'
        }
    }
    
    # UI Configuration
    UI_CONFIG = {
        'theme': {
            'primary_color': '#3498db',
            'secondary_color': '#2980b9',
            'success_color': '#27ae60',
            'warning_color': '#f39c12',
            'error_color': '#e74c3c',
            'background_gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
        },
        'animations': {
            'search_duration': 300,
            'fade_duration': 500,
            'slide_duration': 300
        },
        'performance': {
            'max_search_history': 10,
            'status_refresh_interval': 30000,
            'request_timeout': 30000
        }
    }
    
    # Performance monitoring
    PERFORMANCE_CONFIG = {
        'track_search_times': True,
        'track_error_rates': True,
        'track_service_availability': True,
        'max_metrics_history': 100
    }
    
    # Feature flags
    FEATURE_FLAGS = {
        'enable_real_time_suggestions': True,
        'enable_search_history': True,
        'enable_user_preferences': True,
        'enable_performance_monitoring': True,
        'enable_error_tracking': True,
        'enable_service_health_check': True
    }

class DevelopmentConfig(ServiceConfig):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Override service URLs for development
    SERVICES = {
        **ServiceConfig.SERVICES,
        'main': 'http://localhost:8000',
        'topic_detection': 'http://localhost:8006',
        'query_suggestion': 'http://localhost:8010',
        'vector_store': 'http://localhost:8008',
        'embedding_vector_store': 'http://localhost:8008'
    }

class ProductionConfig(ServiceConfig):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Override service URLs for production
    SERVICES = {
        **ServiceConfig.SERVICES,
        'main': os.getenv('MAIN_SERVICE_URL', 'http://localhost:8000'),
        'topic_detection': os.getenv('TOPIC_DETECTION_URL', 'http://localhost:8006'),
        'query_suggestion': os.getenv('QUERY_SUGGESTION_URL', 'http://localhost:8010'),
        'vector_store': os.getenv('VECTOR_STORE_URL', 'http://localhost:8008'),
        'embedding_vector_store': os.getenv('EMBEDDING_VECTOR_STORE_URL', 'http://localhost:8008')
    }

class TestingConfig(ServiceConfig):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    
    # Override service URLs for testing
    SERVICES = {
        **ServiceConfig.SERVICES,
        'main': 'http://localhost:8000',
        'topic_detection': 'http://localhost:8006',
        'query_suggestion': 'http://localhost:8010',
        'vector_store': 'http://localhost:8008',
        'embedding_vector_store': 'http://localhost:8008'
    }

# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env: str = None) -> ServiceConfig:
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development') or 'development'
    
    return config_map.get(env, config_map['default'])

# Export current configuration
Config = get_config() 