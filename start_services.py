#!/usr/bin/env python3
"""
Startup script for IR System services
Runs all services on their specified ports
"""

import subprocess
import sys
import os
import time
from typing import List, Dict

# Service configuration with updated ports
SERVICES = {
    'preprocessing_service': {
        'port': 8002,
        'path': 'services.preprocessing_service.main:app',
        'name': 'Preprocessing Service'
    },
    'tfidf_service': {
        'port': 8003,
        'path': 'services.tfidf_service.main:app',
        'name': 'TF-IDF Service'
    },
    'embedding_service': {
        'port': 8004,
        'path': 'services.embedding_service.main:app',
        'name': 'Embedding Service'
    },
    'hybrid_service': {
        'port': 8005,
        'path': 'services.hybrid_service.main:app',
        'name': 'Hybrid Service'
    },
    'topic_detection_service': {
        'port': 8006,
        'path': 'services.topic_detection_service.main:app',
        'name': 'Topic Detection Service'
    },
    'vector_store_service': {
        'port': 8008,
        'path': 'services.vector_store_service.main:app',
        'name': 'Vector Store Service'
    },
    'query_suggestion_service': {
        'port': 8010,
        'path': 'services.query_suggestion_service.main:app',
        'name': 'Query Suggestion Service'
    },
    'api_gateway': {
        'port': 8000,
        'path': 'services.api_gateway.main:app',
        'name': 'API Gateway'
    }
}

def start_service(service_name: str, service_config: Dict) -> subprocess.Popen:
    """Start a single service"""
    print(f"Starting {service_config['name']} on port {service_config['port']}...")
    
    cmd = [
        'uvicorn',
        service_config['path'],
        '--host', '0.0.0.0',
        '--port', str(service_config['port']),
        '--reload'
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except Exception as e:
        print(f"Error starting {service_name}: {e}")
        return None

def main():
    """Main function to start all services"""
    print("🚀 Starting IR System Services...")
    print("=" * 50)
    
    processes = []
    
    # Start all services except API Gateway first
    for service_name, config in SERVICES.items():
        if service_name != 'api_gateway':
            process = start_service(service_name, config)
            if process:
                processes.append((service_name, process))
                time.sleep(1)  # Give each service time to start
    
    # Wait a bit for services to initialize
    print("\n⏳ Waiting for services to initialize...")
    time.sleep(5)
    
    # Start API Gateway last
    gateway_process = start_service('api_gateway', SERVICES['api_gateway'])
    if gateway_process:
        processes.append(('api_gateway', gateway_process))
    
    print("\n✅ All services started successfully!")
    print("=" * 50)
    print("Service Status:")
    for service_name, config in SERVICES.items():
        print(f"  • {config['name']}: http://localhost:{config['port']}")
    
    print("\n🌐 Main Interface: http://localhost:8000")
    print("Press Ctrl+C to stop all services")
    
    try:
        # Wait for all processes
        for service_name, process in processes:
            if process:
                process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        for service_name, process in processes:
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()
        print("All services stopped.")

if __name__ == "__main__":
    main() 