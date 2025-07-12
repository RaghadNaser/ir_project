# offline/argsme/eval_tfidf_argsme.py

import os
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

TFIDF_DIR = "data/vectors/argsme/tfidf"
QUERIES_PATH = "data/raw/argsme/ARGSME_original_queries.tsv"
QRELS_PATH = "data/raw/argsme/ARGSME_original_qrels.tsv"

vectorizer = joblib.load(os.path.join(TFIDF_DIR, "argsme_tfidf_vectorizer.joblib"))
tfidf_matrix = joblib.load(os.path.join(TFIDF_DIR, "argsme_tfidf_matrix.joblib"))
doc_ids = joblib.load(os.path.join(TFIDF_DIR, "argsme_doc_mapping.joblib"))

queries_df = pd.read_csv(QUERIES_PATH, sep="\t")
qrels_df = pd.read_csv(QRELS_PATH, sep="\t")

qrels_dict = defaultdict(set)
for _, row in qrels_df.iterrows():
    if row["relevance"] > 0:
        qrels_dict[str(row["query_id"])].add(str(row["doc_id"]))

aps = []
for _, q in queries_df.iterrows():
    parts = []
    for col in ["text", "description", "title"]:
        if col in q.index:
            value = q.at[col]
            if not pd.isna(value):
                parts.append(str(value))
    query_text = " ".join(parts)
    q_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    ranked = np.argsort(-sims)
    retrieved = [doc_ids[i] for i in ranked[:1000]]
    relevant = qrels_dict.get(str(q["query_id"]), set())
    if not relevant:
        continue
    hits = 0
    sum_prec = 0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    ap = sum_prec / len(relevant) if relevant else 0
    aps.append(ap)
print(f"MAP for argsme TF-IDF: {np.mean(aps):.4f}")