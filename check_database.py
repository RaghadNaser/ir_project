#!/usr/bin/env python3
"""
Check database structure and content
"""

import sqlite3
import os

DB_PATH = "data/ir_database_combined.db"

def check_database():
    """Check database structure and content"""
    print("Checking database...")
    
    # Check if database file exists
    if not os.path.exists(DB_PATH):
        print(f"Database file not found: {DB_PATH}")
        return
    
    print(f"Database file found: {DB_PATH}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check available tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        print(f"Available tables: {table_names}")
        
        # Check argsme_raw table
        if "argsme_raw" in table_names:
            cursor.execute("SELECT COUNT(*) FROM argsme_raw")
            argsme_count = cursor.fetchone()[0]
            print(f"argsme_raw table: {argsme_count} rows")
            
            if argsme_count > 0:
                cursor.execute("SELECT doc_id, conclusion, premises_texts, source_title, topic FROM argsme_raw LIMIT 3")
                sample_rows = cursor.fetchall()
                print("Sample argsme_raw rows:")
                for i, row in enumerate(sample_rows):
                    print(f"  Row {i+1}: doc_id={row[0]}, conclusion_length={len(str(row[1])) if row[1] else 0}")
        else:
            print("argsme_raw table not found")
        
        # Check wikir_docs table
        if "wikir_docs" in table_names:
            cursor.execute("SELECT COUNT(*) FROM wikir_docs")
            wikir_count = cursor.fetchone()[0]
            print(f"wikir_docs table: {wikir_count} rows")
            
            if wikir_count > 0:
                cursor.execute("SELECT doc_id, text FROM wikir_docs LIMIT 3")
                sample_rows = cursor.fetchall()
                print("Sample wikir_docs rows:")
                for i, row in enumerate(sample_rows):
                    print(f"  Row {i+1}: doc_id={row[0]}, text_length={len(str(row[1])) if row[1] else 0}")
        else:
            print("wikir_docs table not found")
        
        conn.close()
        
    except Exception as e:
        print(f"Error checking database: {str(e)}")

if __name__ == "__main__":
    check_database() 