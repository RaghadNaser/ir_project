import requests
import json
import time
import sqlite3
from pathlib import Path

def check_database():
    """Check if the database exists and has the required tables"""
    db_path = "data/ir_database_combined.db"
    
    print("Checking Database...")
    print("=" * 40)
    
    if not Path(db_path).exists():
        print(f"❌ Database file not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check available tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        print(f"✅ Database found: {db_path}")
        print(f"📋 Available tables: {table_names}")
        
        # Check table contents
        for table in table_names:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   • {table}: {count:,} rows")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error checking database: {str(e)}")
        return False

def test_agent_service():
    """Test the agent service with various queries"""
    url = "http://127.0.0.1:8011/chat"
    
    test_queries = [
        "information retrieval systems",
        "machine learning algorithms", 
        "artificial intelligence",
        "data mining techniques",
        "search engine optimization",
        "natural language processing",
        "deep learning models",
        "computer vision applications"
    ]
    
    print("\nTesting Agent Service with Database...")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: Query = '{query}'")
        print("-" * 40)
        
        test_data = {
            "message": query,
            "user_id": "test_user"
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=test_data, timeout=30)
            end_time = time.time()
            
            print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
            print(f"📊 Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"💬 Response: {result['response']}")
                print(f"🎯 Confidence: {result['confidence']:.3f}")
                print(f"📄 Documents found: {len(result['documents'])}")
                
                # Show document details
                for j, doc in enumerate(result['documents'][:3], 1):  # Show first 3
                    print(f"\n📋 Document {j}:")
                    print(f"   ID: {doc['id']}")
                    print(f"   Title: {doc['title']}")
                    print(f"   Source: {doc['source']}")
                    print(f"   Score: {doc['score']:.3f}")
                    print(f"   Topic: {doc['topic']}")
                    print(f"   Content: {doc['content'][:100]}...")
                    
            else:
                print(f"❌ Error response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def test_api_gateway():
    """Test the agent service through API Gateway"""
    url = "http://127.0.0.1:8000/agent/chat"
    
    test_data = {
        "message": "information retrieval and machine learning",
        "user_id": "test_user"
    }
    
    print("\n" + "=" * 50)
    print("Testing Agent Service through API Gateway...")
    print("=" * 50)
    print(f"🌐 URL: {url}")
    print(f"🔍 Query: {test_data['message']}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=test_data, timeout=60)
        end_time = time.time()
        
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n💬 Response: {result['response']}")
            print(f"🎯 Confidence: {result['confidence']:.3f}")
            print(f"📄 Documents found: {len(result['documents'])}")
            
            # Count by source
            argsme_count = len([d for d in result['documents'] if d['source'] == 'argsme'])
            wikir_count = len([d for d in result['documents'] if d['source'] == 'wikir'])
            print(f"📊 Sources: ARGSME={argsme_count}, WIKIR={wikir_count}")
            
            # Show top documents
            for i, doc in enumerate(result['documents'][:3], 1):
                print(f"\n📋 Top Document {i}:")
                print(f"   ID: {doc['id']}")
                print(f"   Title: {doc['title']}")
                print(f"   Source: {doc['source']}")
                print(f"   Score: {doc['score']:.3f}")
                print(f"   Content: {doc['content'][:150]}...")
                
        else:
            print(f"❌ Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_extraction_endpoint():
    """Test the extraction endpoint"""
    url = "http://127.0.0.1:8011/extract"
    
    test_data = {
        "message": "artificial intelligence and deep learning",
        "user_id": "test_user"
    }
    
    print("\n" + "=" * 50)
    print("Testing Extraction Endpoint...")
    print("=" * 50)
    print(f"🔍 Query: {test_data['message']}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=test_data, timeout=30)
        end_time = time.time()
        
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📄 Extracted content: {result['extracted_content']}")
            print(f"📋 Summary: {result['summary']}")
            print(f"📊 Sources: {result['sources']}")
            print(f"📄 Total documents: {len(result['documents'])}")
            
        else:
            print(f"❌ Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def check_service_health():
    """Check if the agent service is healthy"""
    url = "http://127.0.0.1:8011/health"
    
    print("Checking Agent Service Health...")
    print("-" * 30)
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Status: {health['status']}")
            print(f"🔗 Database connected: {health['database_connected']}")
            return health['database_connected']
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Database-Based Agent Service Test")
    print("=" * 50)
    
    # First check database
    db_ok = check_database()
    
    if not db_ok:
        print("\n❌ Database check failed. Please ensure the database file exists.")
        print("Expected path: data/ir_database_combined.db")
        exit(1)
    
    # Check service health
    service_ok = check_service_health()
    
    if service_ok:
        print("\n✅ Service is healthy, proceeding with tests...")
        test_agent_service()
        test_api_gateway()
        test_extraction_endpoint()
    else:
        print("\n❌ Service is not healthy. Please start the agent service first.")
        print("Run: python start_agent_real_docs.py") 