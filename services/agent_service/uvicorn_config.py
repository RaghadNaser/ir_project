#!/usr/bin/env python3
"""
Uvicorn configuration for agent service
Optimized to avoid unnecessary model reloading
"""

import uvicorn
from uvicorn.config import Config
from uvicorn.server import Server

def create_optimized_server():
    """Create an optimized uvicorn server configuration"""
    
    config = Config(
        app="services.agent_service.main:app",
        host="0.0.0.0",
        port=8011,
        reload=True,
        reload_dirs=["services/agent_service"],  # Only watch agent service directory
        reload_excludes=["*.pyc", "*.pyo", "__pycache__", "*.log"],
        log_level="info",
        access_log=True,
        workers=1,  # Single worker to avoid model duplication
        loop="asyncio",
        http="httptools",
        ws="websockets",
        lifespan="on",
        env_file=None,
        use_colors=True,
        proxy_headers=True,
        forwarded_allow_ips="*",
        date_header=True,
        server_header=True,
        h11_max_incomplete_event_size=16384,
    )
    
    return Server(config=config)

if __name__ == "__main__":
    server = create_optimized_server()
    server.run() 