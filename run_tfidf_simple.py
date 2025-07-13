#!/usr/bin/env python3
"""
Run Simplified TF-IDF Service
This script runs only the simplified TF-IDF service to avoid version compatibility issues
"""

import subprocess
import sys
import os

def main():
    """Run the simplified TF-IDF service"""
    
    print("🚀 Starting Simplified TF-IDF Service")
    print("=" * 50)
    print("This service loads documents from the database and creates fresh TF-IDF models")
    print("This avoids version compatibility issues with pre-trained models")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("services/tfidf_service/main_simple.py"):
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if database exists
    if not os.path.exists("data/ir_database_combined.db"):
        print("Error: Database not found at data/ir_database_combined.db")
        print("Please make sure the database exists and contains the required tables")
        sys.exit(1)
    
    print("✅ Database found")
    print("✅ Service script found")
    print("\nStarting service on http://localhost:8002")
    print("Press Ctrl+C to stop the service")
    print("-" * 50)
    
    try:
        # Start the service
        process = subprocess.run([
            sys.executable, "services/tfidf_service/main_simple.py"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Service stopped by user")
    except Exception as e:
        print(f"❌ Error running service: {e}")

if __name__ == "__main__":
    main() 