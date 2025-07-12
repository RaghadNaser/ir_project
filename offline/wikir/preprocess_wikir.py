# offline/wikir/preprocess_wikir.py

import os
import pandas as pd
import nltk
import string
from tqdm import tqdm

nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
punctuation_table = str.maketrans("", "", string.punctuation)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(punctuation_table)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

RAW_DIR = "data/raw/wikir"
PROCESSED_DIR = "data/vectors/wikir/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

docs_df = pd.read_csv(os.path.join(RAW_DIR, "documents.tsv"), sep="\t")
tqdm.pandas(desc="Cleaning documents")
docs_df["clean_text"] = docs_df["text"].progress_apply(clean_text)
docs_df = docs_df[["doc_id", "clean_text"]].rename({"clean_text": "text"}, axis=1)
docs_df.to_csv(os.path.join(PROCESSED_DIR, "documents_cleaned.tsv"), sep="\t", index=False)
print("✅ Cleaned WIKIR docs saved.")