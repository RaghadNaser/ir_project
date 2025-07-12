#!/usr/bin/env python3
"""
CORS Configuration Test Script
Tests if all services have proper CORS headers configured
"""

import requests
import sys
import time
from typing import Dict, List, Tuple

# Service configuration with updated ports
SERVICES = {
    'API Gateway': 'http://localhost:8000',
    'Preprocessing Service': 'http://localhost:8002',
    'TF-IDF Service': 'http://localhost:8003',
    'Embedding Service': 'http://localhost:8004',
    'Hybrid Service': 'http://localhost:8005',
    'Topic Detection Service': 'http://localhost:8006',
    'Vector Store Service': 'http://localhost:8008',
    'Query Suggestion Service': 'http://localhost:8010'
}

def test_cors_preflight(service_name: str, service_url: str) -> Tuple[bool, str]:
    """Test CORS preflight request for a service"""
    try:
        # Send OPTIONS request (preflight)
        response = requests.options(
            f"{service_url}/health",
            headers={
                'Origin': 'http://localhost:8000',
                'Access-Control-Request-Method': 'GET',
                'Access-Control-Request-Headers': 'Content-Type'
            },
            timeout=5
        )
        
        # Check if CORS headers are present
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers'),
            'Access-Control-Allow-Credentials': response.headers.get('Access-Control-Allow-Credentials')
        }
        
        # Check if essential CORS headers are present
        if cors_headers['Access-Control-Allow-Origin']:
            return True, f"✅ CORS properly configured: {cors_headers['Access-Control-Allow-Origin']}"
        else:
            return False, "❌ Missing Access-Control-Allow-Origin header"
    
    except requests.exceptions.ConnectionRefused:
        return False, "🔴 Service not running"
    except requests.exceptions.Timeout:
        return False, "⏱️ Service timeout"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def test_actual_request(service_name: str, service_url: str) -> Tuple[bool, str]:
    """Test actual request with CORS headers"""
    try:
        # Send actual GET request
        response = requests.get(
            f"{service_url}/health",
            headers={
                'Origin': 'http://localhost:8000'
            },
            timeout=5
        )
        
        if response.status_code == 200:
            cors_origin = response.headers.get('Access-Control-Allow-Origin')
            if cors_origin:
                return True, f"✅ Service running with CORS: {cors_origin}"
            else:
                return False, "❌ Service running but no CORS headers"
        else:
            return False, f"❌ Service returned status {response.status_code}"
    
    except requests.exceptions.ConnectionRefused:
        return False, "🔴 Service not running"
    except requests.exceptions.Timeout:
        return False, "⏱️ Service timeout"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def main():
    """Main test function"""
    print("🧪 Testing CORS Configuration for All Services")
    print("=" * 60)
    
    all_passed = True
    results = []
    
    for service_name, service_url in SERVICES.items():
        print(f"\n🔍 Testing {service_name} ({service_url})...")
        
        # Test preflight request
        preflight_passed, preflight_msg = test_cors_preflight(service_name, service_url)
        print(f"   Preflight: {preflight_msg}")
        
        # Test actual request
        actual_passed, actual_msg = test_actual_request(service_name, service_url)
        print(f"   Actual:    {actual_msg}")
        
        # Overall result for this service
        service_passed = preflight_passed and actual_passed
        if not service_passed:
            all_passed = False
        
        results.append({
            'service': service_name,
            'url': service_url,
            'preflight': preflight_passed,
            'actual': actual_passed,
            'overall': service_passed
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 CORS Configuration Summary")
    print("=" * 60)
    
    for result in results:
        status = "✅ PASS" if result['overall'] else "❌ FAIL"
        print(f"{status} {result['service']:<25} {result['url']}")
    
    print(f"\n🎯 Overall Result: {'✅ ALL SERVICES CONFIGURED' if all_passed else '❌ SOME SERVICES NEED FIXING'}")
    
    if not all_passed:
        print("\n🔧 Troubleshooting:")
        print("1. Make sure all services are running on their specified ports")
        print("2. Check that CORS middleware is properly added to each service")
        print("3. Verify that allow_origins includes '*' or the specific origin")
        print("4. Restart services after adding CORS configuration")
        
        print("\n📋 Services that need attention:")
        for result in results:
            if not result['overall']:
                print(f"   - {result['service']} ({result['url']})")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 