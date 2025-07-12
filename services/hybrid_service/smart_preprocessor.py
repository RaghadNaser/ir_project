# services/hybrid_service/smart_preprocessor.py

import re
import pandas as pd

def smart_preprocessor(text):
    """Optimized preprocessing for already cleaned ARGSME data"""
    if not isinstance(text, str) or pd.isna(text):
        return ""

    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = re.sub(r'[^\w\s\-]', ' ', text)  # Remove all non-word characters except hyphens
    return text.lower().strip()
