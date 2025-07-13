import requests
import json
import time

def test_agent_service_real_docs():
    """Test the agent service with real document search"""
    url = "http://127.0.0.1:8011/chat"
    
    test_queries = [
        "information retrieval",
        "search algorithms", 
        "machine learning",
        "artificial intelligence",
        "data mining"
    ]
    
    print("Testing Agent Service with Real Documents...")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: Query = '{query}'")
        print("-" * 30)
        
        test_data = {
            "message": query,
            "user_id": "test_user"
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=test_data, timeout=30)
            end_time = time.time()
            
            print(f"Response time: {end_time - start_time:.2f} seconds")
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result['response']}")
                print(f"Confidence: {result['confidence']}")
                print(f"Documents found: {len(result['documents'])}")
                
                for j, doc in enumerate(result['documents'], 1):
                    print(f"\nDocument {j}:")
                    print(f"  ID: {doc['id']}")
                    print(f"  Title: {doc['title']}")
                    print(f"  Score: {doc['score']:.3f}")
                    print(f"  Source: {doc['source']}")
                    print(f"  Content preview: {doc['content'][:100]}...")
                    
            else:
                print(f"Error response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("Request timed out")
        except Exception as e:
            print(f"Error: {str(e)}")

def test_api_gateway_real_docs():
    """Test the agent service through API Gateway with real documents"""
    url = "http://127.0.0.1:8000/agent/chat"
    
    test_data = {
        "message": "information retrieval systems",
        "user_id": "test_user"
    }
    
    print("\n" + "=" * 50)
    print("Testing Agent Service through API Gateway...")
    print("=" * 50)
    print(f"URL: {url}")
    print(f"Query: {test_data['message']}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=test_data, timeout=60)
        end_time = time.time()
        
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nResponse: {result['response']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Documents found: {len(result['documents'])}")
            
            for i, doc in enumerate(result['documents'], 1):
                print(f"\nDocument {i}:")
                print(f"  ID: {doc['id']}")
                print(f"  Title: {doc['title']}")
                print(f"  Score: {doc['score']:.3f}")
                print(f"  Source: {doc['source']}")
                print(f"  Content preview: {doc['content'][:150]}...")
                
        else:
            print(f"Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("Request timed out")
    except Exception as e:
        print(f"Error: {str(e)}")

def check_service_health():
    """Check if the agent service is healthy and documents are loaded"""
    url = "http://127.0.0.1:8011/health"
    
    print("Checking Agent Service Health...")
    print("-" * 30)
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"Status: {health['status']}")
            print(f"Documents loaded: {health['documents_loaded']}")
            return health['documents_loaded']
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {str(e)}")
        return False

if __name__ == "__main__":
    # First check if documents are loaded
    docs_loaded = check_service_health()
    
    if docs_loaded:
        print("\n✅ Documents are loaded, proceeding with tests...")
        test_agent_service_real_docs()
        test_api_gateway_real_docs()
    else:
        print("\n❌ Documents are not loaded. Please check the agent service logs.")
        print("Make sure the data file exists at: data/vectors/argsme/processed/ARGSME_cleaned_docs.tsv") 