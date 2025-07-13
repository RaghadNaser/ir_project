#!/usr/bin/env python3
"""
Test script for the fixed agent service
"""

import requests
import json
import time

def test_agent_service():
    """Test the agent service with different languages and search methods"""
    
    base_url = "http://localhost:8011"
    
    # Test cases
    test_cases = [
        {
            "name": "Arabic Query with Hybrid Search",
            "data": {
                "message": "ابحث عن الذكاء الاصطناعي",
                "language": "ar",
                "search_method": "hybrid",
                "dataset": "argsme",
                "top_k": 3
            }
        },
        {
            "name": "English Query with Hybrid Search",
            "data": {
                "message": "Explain machine learning concepts",
                "language": "en",
                "search_method": "hybrid",
                "dataset": "argsme",
                "top_k": 3
            }
        },
        {
            "name": "English Query with TF-IDF Search",
            "data": {
                "message": "Compare different search methods",
                "language": "en",
                "search_method": "tfidf",
                "dataset": "argsme",
                "top_k": 3
            }
        },
        {
            "name": "English Query with Embedding Search",
            "data": {
                "message": "Find information about climate change",
                "language": "en",
                "search_method": "embedding",
                "dataset": "argsme",
                "top_k": 3
            }
        }
    ]
    
    print("🧪 Testing Agent Service with Fixed Configuration")
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
                print(f"📝 Response: {result.get('response', 'No response')[:100]}...")
                print(f"📊 Confidence: {result.get('confidence', 0):.2f}")
                print(f"🌐 Language: {result.get('language', 'Unknown')}")
                print(f"🔍 Search Method: {result.get('search_method', 'Unknown')}")
                print(f"📄 Documents Found: {len(result.get('documents', []))}")
                
                # Show first document if available
                if result.get('documents'):
                    first_doc = result['documents'][0]
                    print(f"📋 First Document: {first_doc.get('title', 'No title')[:50]}...")
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

if __name__ == "__main__":
    test_agent_service() 