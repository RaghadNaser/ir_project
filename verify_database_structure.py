#!/usr/bin/env python3
"""
Verify database structure for agent service
"""

import sqlite3
import pandas as pd
import os

def verify_database_structure():
    """Verify that the database has the expected structure"""
    
    DB_PATH = "data/ir_database_combined.db"
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Database file not found: {DB_PATH}")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        print("🔍 Verifying database structure...")
        print("=" * 50)
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        print(f"📋 Available tables: {tables}")
        
        # Check ARGSME table structure
        print("\n📊 Checking ARGSME table...")
        if 'argsme_raw' in tables:
            cursor.execute("PRAGMA table_info(argsme_raw);")
            argsme_columns = [col[1] for col in cursor.fetchall()]
            print(f"   Columns: {argsme_columns}")
            
            # Check if required columns exist
            required_argsme = ['doc_id', 'conclusion', 'premises_texts', 'source_title', 'topic', 'acquisition']
            missing_argsme = [col for col in required_argsme if col not in argsme_columns]
            if missing_argsme:
                print(f"   ❌ Missing columns: {missing_argsme}")
            else:
                print("   ✅ All required columns present")
                
            # Check sample data
            cursor.execute("SELECT COUNT(*) FROM argsme_raw;")
            count = cursor.fetchone()[0]
            print(f"   📄 Document count: {count}")
            
            if count > 0:
                cursor.execute("SELECT * FROM argsme_raw LIMIT 1;")
                sample = cursor.fetchone()
                print(f"   📝 Sample doc_id: {sample[0]}")
        else:
            print("   ❌ argsme_raw table not found")
        
        # Check WIKIR table structure
        print("\n📊 Checking WIKIR table...")
        if 'wikir_docs' in tables:
            cursor.execute("PRAGMA table_info(wikir_docs);")
            wikir_columns = [col[1] for col in cursor.fetchall()]
            print(f"   Columns: {wikir_columns}")
            
            # Check if required columns exist
            required_wikir = ['doc_id', 'title', 'text']
            missing_wikir = [col for col in required_wikir if col not in wikir_columns]
            if missing_wikir:
                print(f"   ❌ Missing columns: {missing_wikir}")
            else:
                print("   ✅ All required columns present")
                
            # Check sample data
            cursor.execute("SELECT COUNT(*) FROM wikir_docs;")
            count = cursor.fetchone()[0]
            print(f"   📄 Document count: {count}")
            
            if count > 0:
                cursor.execute("SELECT * FROM wikir_docs LIMIT 1;")
                sample = cursor.fetchone()
                print(f"   📝 Sample doc_id: {sample[0]}")
        else:
            print("   ❌ wikir_docs table not found")
        
        # Test queries that the agent service uses
        print("\n🧪 Testing agent service queries...")
        
        # Test ARGSME query
        try:
            cursor.execute("""
                SELECT doc_id, conclusion, premises_texts, source_title, topic, acquisition
                FROM argsme_raw 
                WHERE doc_id = ?
            """, ("test_id",))
            print("   ✅ ARGSME query syntax is correct")
        except Exception as e:
            print(f"   ❌ ARGSME query failed: {e}")
        
        # Test WIKIR query
        try:
            cursor.execute("""
                SELECT doc_id, title, text
                FROM wikir_docs
                WHERE doc_id = ?
            """, ("test_id",))
            print("   ✅ WIKIR query syntax is correct")
        except Exception as e:
            print(f"   ❌ WIKIR query failed: {e}")
        
        # Test actual data retrieval
        print("\n🔍 Testing actual data retrieval...")
        
        # Get a real ARGSME document
        cursor.execute("SELECT doc_id FROM argsme_raw LIMIT 1;")
        argsme_sample = cursor.fetchone()
        if argsme_sample:
            doc_id = argsme_sample[0]
            cursor.execute("""
                SELECT doc_id, conclusion, premises_texts, source_title, topic, acquisition
                FROM argsme_raw 
                WHERE doc_id = ?
            """, (doc_id,))
            result = cursor.fetchone()
            if result:
                print(f"   ✅ ARGSME document retrieved: {doc_id}")
                print(f"      Title: {result[3][:50]}...")
            else:
                print(f"   ❌ Failed to retrieve ARGSME document: {doc_id}")
        
        # Get a real WIKIR document
        cursor.execute("SELECT doc_id FROM wikir_docs LIMIT 1;")
        wikir_sample = cursor.fetchone()
        if wikir_sample:
            doc_id = wikir_sample[0]
            cursor.execute("""
                SELECT doc_id, title, text
                FROM wikir_docs
                WHERE doc_id = ?
            """, (doc_id,))
            result = cursor.fetchone()
            if result:
                print(f"   ✅ WIKIR document retrieved: {doc_id}")
                print(f"      Title: {result[1][:50]}...")
            else:
                print(f"   ❌ Failed to retrieve WIKIR document: {doc_id}")
        
        conn.close()
        print("\n" + "=" * 50)
        print("✅ Database structure verification completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying database: {e}")
        return False

if __name__ == "__main__":
    verify_database_structure() 