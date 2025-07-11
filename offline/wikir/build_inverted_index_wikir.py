# offline/wikir/build_inverted_index_wikir_advanced.py

import os
import joblib
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

PROCESSED_PATH = "data/vectors/wikir/processed/documents_cleaned.tsv"
INDEX_PATH = "data/vectors/wikir/inverted_index.joblib"

df = pd.read_csv(PROCESSED_PATH, sep="\t")
doc_ids = df["doc_id"].astype(str).tolist()
texts = df["text"].astype(str).tolist()
total_docs = len(doc_ids)

print(f"Building inverted index for {total_docs:,} documents...")

# بناء TF و TF-IDF
vectorizer = TfidfVectorizer(
    preprocessor=None,
    tokenizer=None,
    analyzer='word',
    lowercase=False
)
tf_matrix = vectorizer.fit_transform(texts)
terms = vectorizer.get_feature_names_out()

# بناء الفهرس المعكوس
inverted_index = defaultdict(dict)
tf_csr = tf_matrix.tocsr()

rows, cols = tf_csr.nonzero()
for row, col in tqdm(zip(rows, cols), total=len(rows), desc="Populating index"):
    doc_id = doc_ids[row]
    term = terms[col]
    inverted_index[term][doc_id] = {
        "tf": tf_csr[row, col]
    }

# إحصائيات الطول
doc_lengths = {doc_id: len(texts[i].split()) for i, doc_id in enumerate(doc_ids)}

# حفظ الفهرس
data_to_save = {
    "index": dict(inverted_index),
    "doc_lengths": doc_lengths,
    "total_docs": total_docs
}
joblib.dump(data_to_save, INDEX_PATH)
print("✅ Index build and save complete.")