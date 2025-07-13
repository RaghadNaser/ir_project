#!/usr/bin/env python3
"""
Test script for the agent service with database fix
"""

import requests
import json
import time

def test_agent_service():
    """Test the agent service with different queries"""
    
    base_url = "http://localhost:8011"
    
    # Test cases
    test_cases = [
        {
            "name": "Arabic Query - Hybrid Search",
            "data": {
                "message": "ابحث عن الذكاء الاصطناعي",
                "language": "ar",
                "search_method": "hybrid",
                "dataset": "argsme",
                "top_k": 3
            }
        },
        {
            "name": "English Query - Hybrid Search",
            "data": {
                "message": "Explain machine learning concepts",
                "language": "en",
                "search_method": "hybrid",
                "dataset": "argsme",
                "top_k": 3
            }
        },
        {
            "name": "English Query - TF-IDF Search",
            "data": {
                "message": "Compare different search methods",
                "language": "en",
                "search_method": "tfidf",
                "dataset": "argsme",
                "top_k": 3
            }
        },
        {
            "name": "English Query - Embedding Search",
            "data": {
                "message": "Find information about climate change",
                "language": "en",
                "search_method": "embedding",
                "dataset": "argsme",
                "top_k": 3
            }
        },
        {
            "name": "WIKIR Dataset Query",
            "data": {
                "message": "artificial intelligence",
                "language": "en",
                "search_method": "hybrid",
                "dataset": "wikir",
                "top_k": 3
            }
        }
    ]
    
    print("🧪 Testing Agent Service with Database Fix")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/chat",
                json=test_case["data"],
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! ({end_time - start_time:.2f}s)")
                print(f"📝 Response: {result.get('response', 'No response')[:150]}...")
                print(f"📊 Confidence: {result.get('confidence', 0):.2f}")
                print(f"🌐 Language: {result.get('language', 'Unknown')}")
                print(f"🔍 Search Method: {result.get('search_method', 'Unknown')}")
                print(f"📄 Documents Found: {len(result.get('documents', []))}")
                
                # Show document details
                if result.get('documents'):
                    print(f"📋 Document Details:")
                    for j, doc in enumerate(result['documents'][:2], 1):  # Show first 2 docs
                        print(f"   {j}. ID: {doc.get('id', 'N/A')}")
                        print(f"      Title: {doc.get('title', 'No title')[:50]}...")
                        print(f"      Source: {doc.get('source', 'Unknown')}")
                        print(f"      Score: {doc.get('score', 0):.3f}")
                else:
                    print("   📭 No documents found")
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"📄 Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ Timeout: Request took too long")
        except requests.exceptions.ConnectionError:
            print("🔌 Connection Error: Could not connect to agent service")
        except Exception as e:
            print(f"💥 Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🏁 Testing completed!")

def test_service_health():
    """Test service health endpoint"""
    print("\n🏥 Testing Service Health")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:8011/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Service Status: {health.get('status', 'Unknown')}")
            print(f"🔗 Database Connected: {health.get('database_connected', False)}")
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    test_service_health()
    test_agent_service() 