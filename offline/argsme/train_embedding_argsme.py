# offline/argsme/train_embedding_argsme.py

import os
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

BASE_PATH = "data/vectors/argsme"
os.makedirs(BASE_PATH, exist_ok=True)
DOCS_PATH = "data/vectors/argsme/processed/ARGSME_cleaned_docs.tsv"

df = pd.read_csv(DOCS_PATH, sep="\t")
texts = df["text"].astype(str).tolist()
doc_ids = df["doc_id"].astype(str).tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

np.save(os.path.join(BASE_PATH, "argsme_bert_embeddings.npy"), embeddings)
joblib.dump({doc_id: idx for idx, doc_id in enumerate(doc_ids)}, os.path.join(BASE_PATH, "argsme_bert_doc_mapping.joblib"))

print(f"✅ Embeddings saved for argsme: {embeddings.shape}")