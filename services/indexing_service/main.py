# services/indexing_service/main.py

from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI(title="Indexing Service")

DATA_PATH = Path("data/vectors")

class IndexRequest(BaseModel):
    dataset: str

@app.post("/load_index")
def load_index(req: IndexRequest):
    """
    Endpoint: /load_index
    Body: {"dataset": "argsme"}
    Returns: {"status": "loaded", "terms": عدد الكلمات في الفهرس}
    """
    index_path = DATA_PATH / req.dataset / "inverted_index.joblib"
    if not index_path.exists():
        return {"error": "Index file not found"}
    index = joblib.load(index_path)
    return {"status": "loaded", "terms": len(index['index'])}