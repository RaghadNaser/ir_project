import os
import joblib
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from tqdm import tqdm
import time

# === CONFIG ===
DATA_PATH = "data/vectors/argsme/processed/ARGSME_cleaned_docs.tsv"
MODEL_DIR = Path("services/topic_detection_service/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_TOPICS = 10
MAX_FEATURES = 5000
RANDOM_STATE = 42
SAMPLE_SIZE = None
N_WORDS = 10  # عدد الكلمات الأعلى لكل موضوع

# === LOAD DATA ===
print(f"[INFO] Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH, sep="\t")
if SAMPLE_SIZE:
    df = df.head(SAMPLE_SIZE)
texts = df["processed_text"].astype(str).tolist()
print(f"[INFO] Loaded {len(texts)} documents.")

# === VECTORIZE ===
print(f"[INFO] Vectorizing texts (max_features={MAX_FEATURES}) ...")
vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words='english')
start_vec = time.time()
X = vectorizer.fit_transform(tqdm(texts, desc="Vectorizing"))
print(f"[INFO] Vectorized shape: {X.shape} (elapsed: {time.time()-start_vec:.2f}s)")

# === TRAIN LDA ===
print(f"[INFO] Training LDA (n_topics={N_TOPICS}) ...")
start_lda = time.time()
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=RANDOM_STATE, verbose=1)
lda.fit(X)
elapsed_lda = time.time() - start_lda
print(f"[INFO] LDA training completed in {elapsed_lda/60:.2f} min.")

# === SAVE MODEL ===
model_path = MODEL_DIR / f"lda_argsme_{N_TOPICS}.joblib"
vect_path = MODEL_DIR / f"vectorizer_argsme_{N_TOPICS}.joblib"
joblib.dump(lda, model_path)
joblib.dump(vectorizer, vect_path)
print(f"[INFO] Model saved to: {model_path}")
print(f"[INFO] Vectorizer saved to: {vect_path}")

# === SHOW TOPICS ===
def print_topics(lda, vectorizer, n_words=10):
    words = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [words[i] for i in top_indices]
        print(f"Topic {idx+1}: {', '.join(top_words)}")

print("\n[INFO] Top words per topic:")
print_topics(lda, vectorizer, n_words=N_WORDS)

print("[INFO] Training complete. You can now use the model in your service.") 