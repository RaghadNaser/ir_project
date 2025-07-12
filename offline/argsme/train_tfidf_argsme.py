# offline/argsme/train_tfidf_argsme.py

import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

PROCESSED_PATH = "data/vectors/argsme/processed/ARGSME_cleaned_docs.tsv"
TFIDF_DIR = "data/vectors/argsme/tfidf"
os.makedirs(TFIDF_DIR, exist_ok=True)

with open('data/vectors/argsme/processed/documents_cleaned.tsv', 'r', encoding='utf-8') as f:
    docs = [line.strip() for line in f]

vectorizer = TfidfVectorizer(
    preprocessor=None,
    tokenizer=None,
    analyzer='word',
    lowercase=False,
    token_pattern=None
)
X = vectorizer.fit_transform(docs)
joblib.dump(vectorizer, os.path.join(TFIDF_DIR, "argsme_tfidf_vectorizer.joblib"))
joblib.dump(X, os.path.join(TFIDF_DIR, "argsme_tfidf_matrix.joblib"))
joblib.dump(doc_ids, os.path.join(TFIDF_DIR, "argsme_doc_mapping.joblib"))

print(f" TF-IDF trained and saved for argsme: {getattr(X, 'shape', type(X))}")