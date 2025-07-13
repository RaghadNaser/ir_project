import requests
import json
import time

def test_agent_service():
    """Test the simplified agent service directly"""
    url = "http://127.0.0.1:8011/chat"
    
    test_data = {
        "message": "information retrieval systems",
        "user_id": "test_user"
    }
    
    print("Testing agent service directly...")
    print(f"URL: {url}")
    print(f"Data: {json.dumps(test_data, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=test_data, timeout=30)
        end_time = time.time()
        
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("Request timed out")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_api_gateway():
    """Test the agent service through API Gateway"""
    url = "http://127.0.0.1:8000/agent/chat"
    
    test_data = {
        "message": "information retrieval systems",
        "user_id": "test_user"
    }
    
    print("\nTesting agent service through API Gateway...")
    print(f"URL: {url}")
    print(f"Data: {json.dumps(test_data, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=test_data, timeout=60)
        end_time = time.time()
        
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("Request timed out")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_agent_service()
    test_api_gateway() 