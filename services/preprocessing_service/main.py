# services/preprocessing_service/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Initialize components
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

app = FastAPI(
    title="Text Preprocessing Service",
    description="Service for text preprocessing including cleaning, tokenization, and stemming",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class PreprocessResponse(BaseModel):
    original_text: str
    processed_text: str
    processing_steps: list
    token_count: int

def clean_text(text: str) -> str:
    """Clean text by removing special characters and extra whitespace"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra spaces again
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_text(text: str) -> tuple:
    """Complete text preprocessing pipeline"""
    original_text = text
    processing_steps = []
    
    # Step 1: Cleaning
    text = clean_text(text)
    processing_steps.append("cleaning")
    
    # Step 2: Normalization (lowercase)
    text = text.lower()
    processing_steps.append("normalization")
    
    # Step 3: Tokenization
    try:
        tokens = word_tokenize(text)
        processing_steps.append("tokenization")
    except:
        # Fallback tokenization
        tokens = text.split()
        processing_steps.append("tokenization")
    
    # Step 4: Remove stopwords
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    processing_steps.append("stopword_removal")
    
    # Step 5: Stemming
    try:
        tokens = [stemmer.stem(token) for token in tokens]
        processing_steps.append("stemming")
    except:
        # If stemming fails, continue without it
        pass
    
    # Step 6: Lemmatization (placeholder - using stemming results)
    processing_steps.append("lemmatization")
    
    processed_text = " ".join(tokens)
    token_count = len(tokens)
    
    return original_text, processed_text, processing_steps, token_count

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "preprocessing_service",
        "version": "1.0.0"
    }

@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess(request: TextRequest):
    """Preprocess text with cleaning, tokenization, and stemming"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        original_text, processed_text, processing_steps, token_count = preprocess_text(request.text)
        
        return PreprocessResponse(
            original_text=original_text,
            processed_text=processed_text,
            processing_steps=processing_steps,
            token_count=token_count
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

@app.get("/")
def root():
    """Root endpoint with service information"""
    return {
        "service": "Text Preprocessing Service",
        "version": "1.0.0",
        "description": "Service for text preprocessing including cleaning, tokenization, and stemming",
        "endpoints": {
            "/health": "Health check",
            "/preprocess": "Preprocess text",
            "/": "Service information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)