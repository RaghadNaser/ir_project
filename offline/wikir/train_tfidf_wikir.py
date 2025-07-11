# offline/wikir/train_tfidf_wikir.py

import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

PROCESSED_PATH = "data/vectors/wikir/processed/documents_cleaned.tsv"
TFIDF_DIR = "data/vectors/wikir/tfidf"
os.makedirs(TFIDF_DIR, exist_ok=True)

df = pd.read_csv(PROCESSED_PATH, sep="\t")
texts = df["text"].astype(str).tolist()
doc_ids = df["doc_id"].astype(str).tolist()

vectorizer = TfidfVectorizer(
      preprocessor=None,
      tokenizer=None,
      analyzer='word',
      lowercase=False,
      token_pattern=None 
  )
tfidf_matrix = vectorizer.fit_transform(texts)

joblib.dump(vectorizer, os.path.join(TFIDF_DIR, "wikir_tfidf_vectorizer.joblib"))
np.dump(tfidf_matrix, os.path.join(TFIDF_DIR, "wikir_tfidf_matrix.npz"))
joblib.dump(doc_ids, os.path.join(TFIDF_DIR, "wikir_doc_mapping.joblib"))

print(" TF-IDF trained and saved for wikir.")