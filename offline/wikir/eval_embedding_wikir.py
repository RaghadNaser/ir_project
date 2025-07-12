# offline/wikir/eval_embedding_wikir.py

import os
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

VECTORS_PATH = "data/vectors/wikir"
QUERIES_PATH = "data/raw/wikir/queries.tsv"
QRELS_PATH = "data/raw/wikir/qrels.tsv"

embeddings = np.load(os.path.join(VECTORS_PATH, "wikir_bert_embeddings.npy"))
doc_mapping = joblib.load(os.path.join(VECTORS_PATH, "wikir_bert_doc_mapping.joblib"))
doc_ids = list(doc_mapping.keys())
id2idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

model = SentenceTransformer("all-MiniLM-L6-v2")

queries_df = pd.read_csv(QUERIES_PATH, sep="\t")
qrels_df = pd.read_csv(QRELS_PATH, sep="\t")

qrels_dict = defaultdict(set)
for _, row in qrels_df.iterrows():
    if row["relevance"] > 0:
        qrels_dict[str(row["query_id"])].add(str(row["doc_id"]))

aps = []
for _, q in queries_df.iterrows():
    query_text = str(q["text"])
    q_emb = model.encode([query_text])
    sims = cosine_similarity(q_emb, embeddings)[0]
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
print(f"MAP for wikir BERT: {np.mean(aps):.4f}")