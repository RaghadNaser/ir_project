#!/usr/bin/env python3
"""
Startup script for all services with correct port configuration
"""

import subprocess
import time
import requests
import sys
import os
from pathlib import Path

# Service configurations
SERVICES = {
    "preprocessing": {
        "port": 8002,
        "command": ["python", "-m", "uvicorn", "services.preprocessing_service.main:app", "--host", "0.0.0.0", "--port", "8002"]
    },
    "tfidf": {
        "port": 8003,
        "command": ["python", "-m", "uvicorn", "services.tfidf_service.main:app", "--host", "0.0.0.0", "--port", "8003"]
    },
    "embedding": {
        "port": 8004,
        "command": ["python", "-m", "uvicorn", "services.embedding_service.main:app", "--host", "0.0.0.0", "--port", "8004"]
    },
    "hybrid": {
        "port": 8005,
        "command": ["python", "-m", "uvicorn", "services.hybrid_service.main:app", "--host", "0.0.0.0", "--port", "8005"]
    },
    "topic_detection": {
        "port": 8006,
        "command": ["python", "-m", "uvicorn", "services.topic_detection_service.main:app", "--host", "0.0.0.0", "--port", "8006"]
    },
    "query_suggestions": {
        "port": 8010,
        "command": ["python", "-m", "uvicorn", "services.query_suggestion_service.main:app", "--host", "0.0.0.0", "--port", "8010"]
    },
    "agent": {
        "port": 8011,
        "command": ["python", "-m", "uvicorn", "services.agent_service.main:app", "--host", "0.0.0.0", "--port", "8011"]
    },
    "api_gateway": {
        "port": 8000,
        "command": ["python", "-m", "uvicorn", "services.api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    }
}

def check_service_health(service_name, port, timeout=30):
    """Check if a service is healthy"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def start_service(service_name, config):
    """Start a service"""
    print(f"🚀 Starting {service_name} service on port {config['port']}...")
    
    try:
        # Start the service
        process = subprocess.Popen(
            config["command"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for the service to start
        time.sleep(3)
        
        # Check if service is healthy
        if check_service_health(service_name, config["port"]):
            print(f"✅ {service_name} service started successfully!")
            return process
        else:
            print(f"⚠️  {service_name} service started but health check failed")
            return process
            
    except Exception as e:
        print(f"❌ Failed to start {service_name} service: {e}")
        return None

def main():
    """Main function to start all services"""
    print("🎯 Starting Information Retrieval System Services")
    print("=" * 60)
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    processes = {}
    
    # Start services in order
    service_order = [
        "preprocessing",
        "tfidf", 
        "embedding",
        "hybrid",
        "topic_detection",
        "query_suggestions",
        "agent",
        "api_gateway"
    ]
    
    for service_name in service_order:
        if service_name in SERVICES:
            config = SERVICES[service_name]
            process = start_service(service_name, config)
            if process:
                processes[service_name] = process
            time.sleep(2)  # Wait between services
    
    print("\n" + "=" * 60)
    print("🎉 All services started!")
    print("\n📋 Service Status:")
    
    # Check all services
    for service_name, config in SERVICES.items():
        status = "✅ Healthy" if check_service_health(service_name, config["port"]) else "❌ Unhealthy"
        print(f"   {service_name}: {status} (http://localhost:{config['port']})")
    
    print(f"\n🌐 Main Interface: http://localhost:8000")
    print(f"🤖 Agent Interface: http://localhost:8000/agent")
    print(f"💬 Chat Interface: http://localhost:8000/chat")
    
    print("\n⏹️  Press Ctrl+C to stop all services")
    
    try:
        # Keep the script running
        while True:
            time.sleep(10)
            
            # Check service health periodically
            print("\n🔍 Health Check:")
            for service_name, config in SERVICES.items():
                if service_name in processes:
                    status = "✅" if check_service_health(service_name, config["port"]) else "❌"
                    print(f"   {service_name}: {status}")
                    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all services...")
        
        # Stop all processes
        for service_name, process in processes.items():
            if process:
                print(f"🛑 Stopping {service_name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        print("✅ All services stopped!")

if __name__ == "__main__":
    main() 