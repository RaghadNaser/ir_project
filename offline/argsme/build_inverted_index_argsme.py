# offline/argsme/build_inverted_index_argsme.py

import os
import pandas as pd
import joblib
from collections import defaultdict

PROCESSED_PATH = "data/vectors/argsme/processed/ARGSME_cleaned_docs.tsv"
INDEX_PATH = "data/vectors/argsme/inverted_index.joblib"

df = pd.read_csv(PROCESSED_PATH, sep="\t")
inverted_index = defaultdict(set)

for idx, row in df.iterrows():
    doc_id = row["doc_id"]
    tokens = str(row["text"]).split()
    for token in set(tokens):
        inverted_index[token].add(doc_id)

inverted_index = {k: list(v) for k, v in inverted_index.items()}
joblib.dump(inverted_index, INDEX_PATH)
print(f" Inverted index built and saved for argsme: {len(inverted_index)} terms")