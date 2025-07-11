#!/usr/bin/env python3
"""
🚀 Service Launcher Script
Run all IR services with separated architecture
"""

import subprocess
import time
import sys
import os
from typing import Dict, List
import requests

# Service configurations
SERVICES = {
    "embedding_only": {
        "path": "services/embedding_service/embedding_only.py",
        "port": 8009,
        "name": "Embedding Only Service",
        "description": "Text → Vector conversion",
    },
    "vector_store": {
        "path": "services/vector_store_service/main.py",
        "port": 8007,
        "name": "Vector Store Service",
        "description": "Vector → Search results",
    },
    "traditional_search": {
        "path": "services/embedding_service/main.py",
        "port": 8004,
        "name": "Traditional Search Service",
        "description": "Text → Search results (complete pipeline)",
    },
    "unified_search": {
        "path": "services/unified_search_service/main.py",
        "port": 8006,
        "name": "Unified Search Service",
        "description": "Unified service with vector store option",
    },
}


def print_banner():
    """Print service banner"""
    print("=" * 70)
    print("🚀 IR SERVICES LAUNCHER")
    print("=" * 70)
    print("Available Services:")
    for key, service in SERVICES.items():
        print(f"  • {service['name']} (Port {service['port']})")
        print(f"    {service['description']}")
    print("=" * 70)


def check_service_health(port: int, service_name: str) -> bool:
    """Check if service is healthy"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def run_service(service_key: str):
    """Run a single service"""
    if service_key not in SERVICES:
        print(f"❌ Service '{service_key}' not found!")
        return None

    service = SERVICES[service_key]

    print(f"🔄 Starting {service['name']} on port {service['port']}...")

    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run the service
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        service["path"].replace("/", ".").replace(".py", "") + ":app",
        "--host",
        "0.0.0.0",
        "--port",
        str(service["port"]),
        "--reload",
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait a bit and check if it started
        time.sleep(3)

        if check_service_health(service["port"], service["name"]):
            print(f"✅ {service['name']} started successfully!")
            print(f"   🌐 URL: http://localhost:{service['port']}")
            return process
        else:
            print(f"⚠️  {service['name']} may not be fully ready yet...")
            return process

    except Exception as e:
        print(f"❌ Failed to start {service['name']}: {e}")
        return None


def demo_usage():
    """Demo the new architecture"""
    print("\n" + "=" * 50)
    print("🎯 USAGE EXAMPLES")
    print("=" * 50)

    print("\n1️⃣ **Embedding Only Service** (Port 8009):")
    print("   curl -X POST http://localhost:8009/embed \\")
    print("        -H 'Content-Type: application/json' \\")
    print('        -d \'{"text":"machine learning algorithms"}\'')

    print("\n2️⃣ **Vector Store Service** (Port 8007):")
    print("   curl -X POST http://localhost:8007/search \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{")
    print('             "dataset":"argsme",')
    print('             "query_vector":[0.1,0.2,...],')
    print('             "top_k":5')
    print("           }'")

    print("\n3️⃣ **Unified Search Service** (Port 8006):")
    print("   # مع Vector Store:")
    print("   curl -X POST http://localhost:8006/search \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{")
    print('             "dataset":"argsme",')
    print('             "query":"machine learning",')
    print('             "use_vector_store":true')
    print("           }'")

    print("\n   # بدون Vector Store:")
    print("   curl -X POST http://localhost:8006/search \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{")
    print('             "dataset":"argsme",')
    print('             "query":"machine learning",')
    print('             "use_vector_store":false')
    print("           }'")

    print("\n4️⃣ **Compare Methods:**")
    print("   curl http://localhost:8006/compare/argsme?query=machine%20learning")


def main():
    """Main function"""
    print_banner()

    if len(sys.argv) < 2:
        print("\n📋 Usage:")
        print("  python run_services.py <service_name>")
        print("  python run_services.py all")
        print("  python run_services.py demo")
        print("\n🔧 Available services:")
        for key in SERVICES.keys():
            print(f"  • {key}")
        return

    command = sys.argv[1].lower()

    if command == "demo":
        demo_usage()
        return

    if command == "all":
        print("\n🔄 Starting all services...")
        processes = []

        for service_key in SERVICES.keys():
            process = run_service(service_key)
            if process:
                processes.append(process)
            time.sleep(2)  # Wait between services

        if processes:
            print(f"\n✅ Started {len(processes)} services!")
            demo_usage()

            print("\n⏳ Press Ctrl+C to stop all services...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping all services...")
                for process in processes:
                    process.terminate()
                print("✅ All services stopped!")

    elif command in SERVICES:
        process = run_service(command)
        if process:
            demo_usage()
            print(f"\n⏳ Press Ctrl+C to stop {SERVICES[command]['name']}...")
            try:
                process.wait()
            except KeyboardInterrupt:
                print(f"\n🛑 Stopping {SERVICES[command]['name']}...")
                process.terminate()
                print("✅ Service stopped!")
    else:
        print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    main()
