#!/usr/bin/env python3
"""
Real Documents Agent Service Startup Script
This script starts the agent service with real document loading from data files.
"""

import uvicorn
import logging
from services.agent_service.main import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Agent Service with Real Documents...")
    logger.info("Loading documents from data/vectors/argsme/processed/ARGSME_cleaned_docs.tsv")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8011,
        reload=False,  # No reload to prevent model loading issues
        log_level="info"
    ) 