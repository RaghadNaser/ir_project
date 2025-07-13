#!/usr/bin/env python3
"""
Simple Agent Service Startup Script
This script starts the agent service without model reloading and with simplified functionality.
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
    logger.info("Starting Simplified Agent Service on port 8011...")
    logger.info("This version only returns documents without file reading")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8011,
        reload=False,  # No reload to prevent model loading issues
        log_level="info"
    ) 