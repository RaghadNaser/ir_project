#!/usr/bin/env python3
"""
Service Startup Script
Starts all IR system services with proper configuration
"""

import subprocess
import time
import sys
import os
from pathlib import Path

# Service configurations
SERVICES = [
    {
        "name": "API Gateway",
        "script": "services/api_gateway/main.py",
        "port": 8001,
        "description": "Main web interface and API gateway"
    },
    {
        "name": "Preprocessing Service",
        "script": "services/preprocessing_service/main.py",
        "port": 8003,
        "description": "Text preprocessing and normalization"
    },
    {
        "name": "TF-IDF Service",
        "script": "services/tfidf_service/main_simple.py",
        "port": 8002,
        "description": "Simplified TF-IDF search (avoids version compatibility issues)"
    },
    {
        "name": "Embedding Service",
        "script": "services/embedding_service/main.py",
        "port": 8004,
        "description": "Neural embedding search"
    },
    {
        "name": "Hybrid Service",
        "script": "services/hybrid_service/main.py",
        "port": 8005,
        "description": "Hybrid search combining TF-IDF and embeddings"
    },
    {
        "name": "Topic Detection Service",
        "script": "services/topic_detection_service/main.py",
        "port": 8006,
        "description": "Topic detection and analysis"
    },
    {
        "name": "Query Suggestions Service",
        "script": "services/query_suggestion_service/main.py",
        "port": 8010,
        "description": "Smart query suggestions and recommendations"
    },
    {
        "name": "Vector Store Service",
        "script": "services/vector_store_service/main.py",
        "port": 8007,
        "description": "Vector database and similarity search"
    },
    {
        "name": "Unified Search Service",
        "script": "services/unified_search_service/main.py",
        "port": 8008,
        "description": "Unified search interface"
    },
    {
        "name": "Indexing Service",
        "script": "services/indexing_service/main.py",
        "port": 8009,
        "description": "Document indexing and management"
    },
    {
        "name": "Agent Service",
        "script": "services/agent_service/run_agent_optimized.py",
        "port": 8011,
        "description": "Professional AI agent with conversational search and document content extraction (Optimized Production Mode - Single Model Loading)"
    }
]

def check_port_available(port):
    """Check if a port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def start_service(service_config):
    """Start a single service"""
    name = service_config["name"]
    script = service_config["script"]
    port = service_config["port"]
    description = service_config["description"]
    
    # Check if script exists
    if not os.path.exists(script):
        print(f"Error: {name}: Script not found at {script}")
        return None
    
    # Check if port is available
    if not check_port_available(port):
        print(f"Warning: {name}: Port {port} is already in use")
        return None

    print(f"Starting {name} on port {port}...")
    print(f"   Description: {description}")
    
    try:
        # Start the service
        process = subprocess.Popen([
            sys.executable, script
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait a moment for the service to start
        time.sleep(2)

        # Check if process is still running
        if process.poll() is None:
            print(f"Success: {name} started successfully (PID: {process.pid})")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"Error: {name} failed to start")
            print(f"   Error: {stderr.decode()}")
            return None

    except Exception as e:
        print(f"Error: {name}: Error starting service - {e}")
        return None

def main():
    """Main function to start all services"""
    print("Information Retrieval System - Service Manager")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("services"):
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Start services
    processes = []

    for service in SERVICES:
        process = start_service(service)
        if process:
            processes.append((service["name"], process))
        time.sleep(1)  # Small delay between services

    print("\n" + "=" * 60)
    print("All services started!")
    print(f"Started {len(processes)} out of {len(SERVICES)} services")
    print("\nAccess the system at: http://localhost:8001")
    print("Agent Service (for testing): http://localhost:8011")
    print("Service Status:")
    
    for name, process in processes:
        print(f"   Success: {name} (PID: {process.pid})")

    print("\nPress Ctrl+C to stop all services")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if any process has died
            for name, process in processes[:]:
                if process.poll() is not None:
                    print(f"Warning: {name} has stopped unexpectedly")
                    processes.remove((name, process))
            
            if not processes:
                print("Error: All services have stopped")
                break
                
    except KeyboardInterrupt:
        print("\nStopping all services...")
        
        for name, process in processes:
            print(f"Stopping {name}...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                print(f"Warning: Error stopping {name}: {e}")

        print("All services stopped")

if __name__ == "__main__":
    main()
