#!/usr/bin/env python3
"""
Database-Based Agent Service Startup Script
This script starts the agent service that connects to the SQLite database
and supports free-form message queries like the hybrid system.
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
    logger.info("🚀 Starting Database-Based Agent Service...")
    logger.info("📊 Connecting to: data/ir_database_combined.db")
    logger.info("🔍 Supporting: ARGSME and WIKIR datasets")
    logger.info("💬 Features: Free-form messages, real document content")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8011,
        reload=False,  # No reload for database connections
        log_level="info"
    ) 