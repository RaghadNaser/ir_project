import os
import joblib
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from tqdm import tqdm
import time

# === CONFIG ===
DATA_PATH = "data/vectors/wikir/processed/documents_cleaned.tsv"
MODEL_DIR = Path("services/topic_detection_service/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_TOPICS = 10  # عدد المواضيع
MAX_FEATURES = 5000  # حجم الـ vocabulary
RANDOM_STATE = 42
SAMPLE_SIZE = None  # ضع رقمًا (مثلاً 5000) لتدريب عينة فقط
N_WORDS = 10  # عدد الكلمات الأعلى لكل موضوع

# === LOAD DATA ===
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH, sep="\t")
if SAMPLE_SIZE is not None:
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
if "processed_text" in df.columns:
    texts = df["processed_text"].astype(str).tolist()
elif "text" in df.columns:
    texts = df["text"].astype(str).tolist()
else:
    raise ValueError("No suitable text column found in the dataframe.")
print(f"Loaded {len(texts)} documents.")

# === VECTORIZE ===
print(f"Vectorizing texts (max_features={MAX_FEATURES}) ...")
vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words='english')
X = vectorizer.fit_transform(tqdm(texts, desc="Vectorizing"))
print(f"Vectorized shape: {X.shape}")

# === TRAIN LDA ===
print(f"Training LDA (n_topics={N_TOPICS}) ...")
start = time.time()
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=RANDOM_STATE)
lda.fit(X)
elapsed = time.time() - start
print(f"LDA training completed in {elapsed/60:.2f} min.")

# === SAVE MODEL ===
model_path = MODEL_DIR / f"lda_wikir_{N_TOPICS}.joblib"
vect_path = MODEL_DIR / f"vectorizer_wikir_{N_TOPICS}.joblib"

joblib.dump(lda, model_path)
joblib.dump(vectorizer, vect_path)

print(f"Model saved to: {model_path}")
print(f"Vectorizer saved to: {vect_path}")

# === SHOW TOPICS ===
def print_topics(lda, vectorizer, n_words=10):
    words = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [words[i] for i in top_indices]
        print(f"Topic {idx+1}: {', '.join(top_words)}")

print("\nTop words per topic:")
print_topics(lda, vectorizer, n_words=N_WORDS) 