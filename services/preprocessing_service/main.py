# services/preprocessing_service/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

app = FastAPI(title="Preprocessing Service")

class TextRequest(BaseModel):
    text: str
    dataset: str = "wikir"  # Dataset name

# Wikir-specific cleaning function
def clean_text_wikir(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

# ARGSME-specific cleaning function
def clean_text_argsme(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

    # Choose cleaning function based on dataset
CLEANERS = {
    "wikir": clean_text_wikir,
    "argsme": clean_text_argsme
}

def clean_text(text, dataset="wikir"):
    cleaner = CLEANERS.get(dataset, clean_text_wikir)
    return cleaner(text)

@app.post("/clean")
def clean_endpoint(req: TextRequest):
    return {"cleaned_text": clean_text(req.text, req.dataset)}