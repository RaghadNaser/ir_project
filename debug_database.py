#!/usr/bin/env python3
"""
Debug script to check database connection and term extraction
"""

import sqlite3
import os
import sys

# Database path
DB_PATH = "data/ir_database_combined.db"

def check_database():
    """Check database connection and tables"""
    print("🔍 Checking database...")
    
    # Check if file exists
    if not os.path.exists(DB_PATH):
        print(f"❌ Database file not found: {DB_PATH}")
        return False
    
    print(f"✅ Database file exists: {DB_PATH}")
    print(f"📊 File size: {os.path.getsize(DB_PATH) / (1024*1024):.2f} MB")
    
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check available tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        print(f"📋 Available tables: {table_names}")
        
        # Check argsme_raw table
        if "argsme_raw" in table_names:
            cursor.execute("SELECT COUNT(*) FROM argsme_raw")
            count = cursor.fetchone()[0]
            print(f"📊 argsme_raw table has {count} rows")
            
            # Check sample data
            cursor.execute("SELECT doc_id, conclusion, premises_texts FROM argsme_raw LIMIT 3")
            sample_rows = cursor.fetchall()
            print("📝 Sample data from argsme_raw:")
            for i, row in enumerate(sample_rows):
                print(f"  Row {i+1}:")
                print(f"    doc_id: {row[0]}")
                print(f"    conclusion: {str(row[1])[:100]}..." if row[1] else "    conclusion: None")
                print(f"    premises_texts: {str(row[2])[:100]}..." if row[2] else "    premises_texts: None")
                print()
        else:
            print("❌ argsme_raw table not found")
        
        # Check wikir_docs table
        if "wikir_docs" in table_names:
            cursor.execute("SELECT COUNT(*) FROM wikir_docs")
            count = cursor.fetchone()[0]
            print(f"📊 wikir_docs table has {count} rows")
            
            # Check sample data
            cursor.execute("SELECT doc_id, text FROM wikir_docs LIMIT 3")
            sample_rows = cursor.fetchall()
            print("📝 Sample data from wikir_docs:")
            for i, row in enumerate(sample_rows):
                print(f"  Row {i+1}:")
                print(f"    doc_id: {row[0]}")
                print(f"    text: {str(row[1])[:100]}..." if row[1] else "    text: None")
                print()
        else:
            print("❌ wikir_docs table not found")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_term_extraction():
    """Test term extraction functionality"""
    print("\n🧪 Testing term extraction...")
    
    try:
        # Import the service
        sys.path.append('services/query_suggestion_service')
        from main import SmartQuerySuggestionService
        
        # Create service instance
        service = SmartQuerySuggestionService()
        
        # Test extraction for argsme
        print("📚 Testing extraction for argsme dataset...")
        terms = service.extract_terms_from_documents("argsme", 100)
        print(f"✅ Extracted {len(terms)} terms from argsme")
        
        if terms:
            print("📝 Sample terms:")
            for i, term in enumerate(terms[:10]):
                print(f"  {i+1}. {term['term']} (score: {term['score']:.4f})")
        
        # Test extraction for wikir
        print("\n📚 Testing extraction for wikir dataset...")
        terms = service.extract_terms_from_documents("wikir", 100)
        print(f"✅ Extracted {len(terms)} terms from wikir")
        
        if terms:
            print("📝 Sample terms:")
            for i, term in enumerate(terms[:10]):
                print(f"  {i+1}. {term['term']} (score: {term['score']:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Term extraction error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_suggestion_methods():
    """Test suggestion methods"""
    print("\n🎯 Testing suggestion methods...")
    
    try:
        # Import the service
        sys.path.append('services/query_suggestion_service')
        from main import SmartQuerySuggestionService
        
        # Create service instance
        service = SmartQuerySuggestionService()
        
        # Test semantic_terms method
        print("🧠 Testing semantic_terms method...")
        suggestions = service.get_semantic_term_suggestions("climate change", "argsme", 5)
        print(f"✅ semantic_terms returned {len(suggestions)} suggestions")
        
        if suggestions:
            print("📝 Sample suggestions:")
            for i, suggestion in enumerate(suggestions[:3]):
                print(f"  {i+1}. {suggestion['query']} (score: {suggestion['score']:.4f})")
        
        # Test hybrid_terms method
        print("\n🔗 Testing hybrid_terms method...")
        suggestions = service.get_hybrid_term_suggestions("climate change", "argsme", 5)
        print(f"✅ hybrid_terms returned {len(suggestions)} suggestions")
        
        if suggestions:
            print("📝 Sample suggestions:")
            for i, suggestion in enumerate(suggestions[:3]):
                print(f"  {i+1}. {suggestion['query']} (score: {suggestion['score']:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Suggestion methods error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🔧 Database and Term Extraction Debug")
    print("=" * 50)
    
    # Check database
    if not check_database():
        print("❌ Database check failed")
        return
    
    # Test term extraction
    if not test_term_extraction():
        print("❌ Term extraction test failed")
        return
    
    # Test suggestion methods
    if not test_suggestion_methods():
        print("❌ Suggestion methods test failed")
        return
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    main() 