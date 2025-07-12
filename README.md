# Professional Information Retrieval System

## Overview

A comprehensive information retrieval system built on Service-Oriented Architecture with advanced AI features including:

### Core Services:
- **TF-IDF Service** (Port 8003): Traditional search using TF-IDF
- **Embedding Service** (Port 8004): Semantic search using Embeddings
- **Hybrid Service** (Port 8005): Hybrid search (TF-IDF + Embeddings)
- **Preprocessing Service** (Port 8002): Text processing
- **Indexing Service** (Port 8001): Document indexing

### Additional Services:
- **Topic Detection Service** (Port 8006): Topic discovery
- **Query Suggestion Service** (Port 8010): Query suggestions
- **Professional AI Agent Service** (Port 8011): Advanced Intelligent Agent

### Support Services:
- **API Gateway** (Port 8000): Main system gateway
- **Vector Store Service** (Port 8007): Vector storage (optional)

## Professional AI Agent Service

### What is Professional AI Agent Service?
Professional AI Agent Service is an advanced intelligent system that provides:
- **Natural conversation** with users in multiple languages
- **Semantic understanding** using advanced AI models
- **Context awareness** with long-term memory
- **Intelligent query refinement** with learning capabilities
- **Multi-language support** (Arabic/English)

### Advanced Features:
1. **Semantic Analysis**: Uses sentence transformers for deep understanding
2. **Language Detection**: Automatic Arabic/English language detection
3. **Long-term Memory**: SQLite-based memory system for user preferences
4. **Advanced Intent Recognition**: Multi-level intent analysis with confidence scoring
5. **Intelligent Query Refinement**: Context-aware query improvement
6. **Real-time Learning**: Continuous improvement from user interactions
7. **Professional UI**: Modern interface with dark mode and advanced indicators

### Technical Capabilities:
- **Sentence Transformers**: Semantic similarity analysis
- **Memory Database**: Persistent user preferences and conversation patterns
- **Confidence Scoring**: Real-time confidence assessment
- **Multi-step Reasoning**: Advanced reasoning with detailed steps
- **WebSocket Support**: Real-time conversation capabilities

### How to Use:

#### Via API:
```bash
# Chat with professional AI agent
curl -X POST http://localhost:8011/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is artificial intelligence?",
    "dataset": "argsme",
    "user_id": "user123",
    "conversation_mode": "interactive",
    "include_topics": true,
    "include_suggestions": true
  }'
```

#### Via Professional Interface:
```
http://localhost:8000/agent
```

## Installation and Setup

### Requirements:
```bash
pip install -r requirements.txt
```

### AI Model Dependencies:
The system automatically downloads required AI models:
- **Sentence Transformers**: `all-MiniLM-L6-v2` for semantic analysis
- **Memory Database**: SQLite for long-term storage

### Running Services:

#### Run all services:
```bash
python run_services.py all
```

#### Run specific service:
```bash
python run_services.py agent  # Run Professional AI Agent Service only
```

#### Run API Gateway:
```bash
python services/api_gateway/main.py
```

## Evaluation

### Core Services Evaluation:
- `evaluate_tfidf.ipynb`: TF-IDF evaluation
- `evaluate_embedding.ipynb`: Embeddings evaluation
- `evaluate_hybrid.ipynb`: Hybrid Search evaluation

### Professional AI Agent Evaluation:
- `evaluate_agent.ipynb`: Advanced agent evaluation with semantic analysis

## Project Structure

```
Final_project_IR/
├── services/
│   ├── agent_service/          # Professional AI Agent Service
│   ├── api_gateway/           # API Gateway
│   ├── embedding_service/     # Embedding Service
│   ├── hybrid_service/        # Hybrid Service
│   ├── tfidf_service/         # TF-IDF Service
│   ├── topic_detection_service/ # Topic Detection
│   └── query_suggestion_service/ # Query Suggestions
├── models/                    # Machine learning models
├── data/                      # Data
├── results/                   # Evaluation results
└── config/                    # System configuration
```

## API Documentation

### Professional AI Agent Service Endpoints:

#### POST /chat
Chat with professional AI agent
```json
{
  "message": "User message",
  "session_id": "Session ID (optional)",
  "user_id": "User ID for memory (optional)",
  "dataset": "argsme|wikir",
  "conversation_mode": "interactive|direct|analytical",
  "max_results": 10,
  "include_topics": true,
  "include_suggestions": true
}
```

Response includes:
- **response**: AI-generated response
- **confidence_score**: Confidence level (0-1)
- **reasoning_steps**: Detailed reasoning process
- **language_detected**: Detected language (en/ar)
- **semantic_analysis**: Whether semantic analysis was used

#### GET /sessions
Get active sessions with advanced statistics

#### DELETE /sessions/{session_id}
Delete specific session

#### GET /health
Check service health with feature status

## Advanced Features

### Semantic Analysis:
- Uses sentence transformers for deep understanding
- Compares user queries with intent patterns
- Provides confidence scores based on semantic similarity

### Language Detection:
- Automatic Arabic/English detection
- Language-specific keyword extraction
- Bilingual response generation

### Long-term Memory:
- Stores user preferences and conversation patterns
- Learns from repeated interactions
- Improves responses over time

### Professional Interface:
- Modern, responsive design
- Dark mode support
- Real-time confidence indicators
- Language detection badges
- Semantic analysis indicators

## Testing

### Postman Collections:
- `IR_System_Complete_Testing.postman_collection.json`: Complete system testing
- `agent_postman_collection.json`: Professional AI Agent testing

### Running Tests:
```bash
# Test Professional AI Agent Service
python -m pytest tests/test_agent.py

# Evaluate Professional AI Agent Service
jupyter notebook evaluate_agent.ipynb
```

## Performance

### Memory Usage:
- Lightweight sentence transformer model (~90MB)
- Efficient SQLite memory storage
- Optimized for real-time responses

### Response Time:
- Average response time: <2 seconds
- Semantic analysis: <500ms
- Memory operations: <100ms

## Contributing

1. Fork the project
2. Create new branch
3. Add features
4. Test changes
5. Create Pull Request

## License

This project is for educational and research purposes.

## Support

For inquiries and technical support, please contact the development team.