import pandas as pd
import sys

def check_data_structure():
    """Check the structure of the ARGSME documents file"""
    try:
        file_path = "data/vectors/argsme/processed/ARGSME_cleaned_docs.tsv"
        
        print(f"Checking file: {file_path}")
        print("=" * 50)
        
        # Read just the header to see columns
        df_header = pd.read_csv(file_path, sep='\t', nrows=0)
        print(f"Columns found: {list(df_header.columns)}")
        print(f"Number of columns: {len(df_header.columns)}")
        
        # Read first few rows to see data structure
        df_sample = pd.read_csv(file_path, sep='\t', nrows=3)
        print(f"\nSample data (first 3 rows):")
        print(df_sample)
        
        # Get total number of rows
        total_rows = len(pd.read_csv(file_path, sep='\t'))
        print(f"\nTotal rows in file: {total_rows}")
        
        return True
        
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return False

if __name__ == "__main__":
    check_data_structure() 