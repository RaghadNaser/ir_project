#!/usr/bin/env python3
"""
Test script for the simplified TF-IDF service
"""

import requests
import json
import time

def test_simple_tfidf_service():
    """Test the simplified TF-IDF service"""
    
    base_url = "http://localhost:8003"
    
    # Test cases
    test_cases = [
        {
            "name": "ARGSME Dataset - English Query",
            "data": {
                "dataset": "argsme",
                "query": "artificial intelligence",
                "top_k": 5
            }
        },
        {
            "name": "ARGSME Dataset - Arabic Query",
            "data": {
                "dataset": "argsme",
                "query": "الذكاء الاصطناعي",
                "top_k": 5
            }
        },
        {
            "name": "WIKIR Dataset - English Query",
            "data": {
                "dataset": "wikir",
                "query": "machine learning",
                "top_k": 5
            }
        },
        {
            "name": "WIKIR Dataset - Technical Query",
            "data": {
                "dataset": "wikir",
                "query": "neural networks",
                "top_k": 5
            }
        }
    ]
    
    print("🧪 Testing Simplified TF-IDF Service")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/search",
                json=test_case["data"],
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! ({end_time - start_time:.2f}s)")
                print(f"📊 Total Documents: {result.get('total_documents', 0)}")
                print(f"🔍 Search Method: {result.get('search_method', 'Unknown')}")
                print(f"📄 Results Found: {len(result.get('results', []))}")
                
                # Show results
                if result.get('results'):
                    print(f"📋 Top Results:")
                    for j, doc in enumerate(result['results'][:3], 1):  # Show first 3
                        print(f"   {j}. Doc ID: {doc.get('doc_id', 'N/A')}")
                        print(f"      Title: {doc.get('title', 'No title')[:60]}...")
                        print(f"      Score: {doc.get('score', 0):.4f}")
                        print(f"      Source: {doc.get('source', 'Unknown')}")
                else:
                    print("   📭 No results found")
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"📄 Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ Timeout: Request took too long")
        except requests.exceptions.ConnectionError:
            print("🔌 Connection Error: Could not connect to TF-IDF service")
        except Exception as e:
            print(f"💥 Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")

def test_service_health():
    """Test service health endpoint"""
    print("\n🏥 Testing TF-IDF Service Health")
    print("-" * 35)
    
    try:
        response = requests.get("http://localhost:8003/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Service Status: {health.get('status', 'Unknown')}")
            print(f"🔧 Service Type: {health.get('service', 'Unknown')}")
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_service_info():
    """Test service info endpoint"""
    print("\nℹ️  Testing TF-IDF Service Info")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:8003/", timeout=10)
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Service: {info.get('service', 'Unknown')}")
            print(f"📊 Status: {info.get('status', 'Unknown')}")
            print(f"📝 Description: {info.get('description', 'No description')}")
        else:
            print(f"❌ Info check failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Info check error: {e}")

if __name__ == "__main__":
    test_service_health()
    test_service_info()
    test_simple_tfidf_service() 