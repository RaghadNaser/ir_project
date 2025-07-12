# offline/argsme/preprocess_argsme.py

import os
import pandas as pd
import nltk
import string
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

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

# المسارات
RAW_PATH = "data/raw/argsme/ARGSME_original_docs.tsv"
PROCESSED_PATH = "data/vectors/argsme/processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)

docs_df = pd.read_csv(RAW_PATH, sep="\t")
tqdm.pandas(desc="Cleaning documents")
docs_df["text"] = docs_df["conclusion"].fillna('') + " " + docs_df["premises_texts"].fillna('')
docs_df["clean_text"] = docs_df["text"].progress_apply(clean_text)
docs_df = docs_df[["doc_id", "clean_text"]].rename({"clean_text": "text"}, axis=1)
docs_df.to_csv(os.path.join(PROCESSED_PATH, "ARGSME_cleaned_docs.tsv"), sep="\t", index=False)
print("✅ Cleaned ARGSME docs saved.")

# حمّل النصوص المنظفة (مثلاً من ملف TSV)
df = pd.read_csv("data/vectors/wikir/processed/documents_cleaned.tsv", sep="\t")
texts = df["text"].astype(str).tolist()

vectorizer = TfidfVectorizer(
    preprocessor=None,
    tokenizer=None,
    analyzer='word',
    lowercase=False
)
tfidf_matrix = vectorizer.fit_transform(texts)

joblib.dump(vectorizer, "data/vectors/wikir/tfidf/wikir_tfidf_vectorizer.joblib")
joblib.dump(tfidf_matrix, "data/vectors/wikir/tfidf/wikir_tfidf_matrix.joblib")