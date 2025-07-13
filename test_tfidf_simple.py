"""
Test script for the simplified TF-IDF service
"""

import requests
import json
import time

def test_tfidf_simple():
    """Test the simplified TF-IDF service"""
    
    # Service URL
    base_url = "http://localhost:8002"
    
    # Test queries
    test_queries = [
        {"dataset": "argsme", "query": "climate change", "top_k": 5},
        {"dataset": "wikir", "query": "artificial intelligence", "top_k": 5},
        {"dataset": "argsme", "query": "vaccination", "top_k": 3},
        {"dataset": "wikir", "query": "machine learning", "top_k": 3}
    ]
    
    print("🧪 Testing Simplified TF-IDF Service")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        print("Make sure the simplified TF-IDF service is running on port 8002")
        return
    
    # Test search endpoints
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {test_case['dataset']} - '{test_case['query']}'")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/search",
                json=test_case,
                timeout=30
            )
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Search successful")
                print(f"   Method: {data.get('search_method', 'unknown')}")
                print(f"   Execution time: {data.get('execution_time', 0):.3f}s")
                print(f"   Request time: {request_time:.3f}s")
                print(f"   Results found: {len(data.get('results', []))}")
                
                # Show top results
                results = data.get('results', [])
                for j, result in enumerate(results[:3], 1):
                    print(f"   {j}. Doc ID: {result.get('doc_id', 'N/A')} (Score: {result.get('score', 0):.4f})")
                
                if len(results) > 3:
                    print(f"   ... and {len(results) - 3} more results")
                    
            else:
                print(f"❌ Search failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"⏰ Request timed out after 30 seconds")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test stats endpoint
    print(f"\n📊 Testing stats endpoint")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Stats retrieved successfully")
            print(f"   Method: {stats.get('method', 'unknown')}")
            print(f"   Description: {stats.get('description', 'unknown')}")
            
            datasets = stats.get('datasets', {})
            for dataset, info in datasets.items():
                print(f"   {dataset}: {info.get('documents_loaded', 0)} docs, {info.get('vocabulary_size', 0)} vocab")
        else:
            print(f"❌ Stats failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Stats error: {e}")
    
    print(f"\n🎉 Testing completed!")

if __name__ == "__main__":
    test_tfidf_simple() 