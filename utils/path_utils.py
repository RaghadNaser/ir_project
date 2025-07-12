from pathlib import Path
from typing import Dict, Any
from config.settings import DATASETS

def get_tfidf_paths(dataset_name: str) -> Dict[str, Path]:
    """
    Get TF-IDF file paths for a specific dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing paths to TF-IDF files
    """
    base_path = Path("data/vectors") / dataset_name / "tfidf"
    
    if dataset_name == "wikir":
        return {
            "vectorizer": base_path / "wikir_tfidf_vectorizer.joblib",
            "matrix": base_path / "wikir_tfidf_matrix.npz",
            "mapping": base_path / "wikir_doc_mapping_fixed.tsv"
        }
    elif dataset_name == "argsme":
        return {
            "vectorizer": base_path / "argsme_tfidf_vectorizer_improved.joblib",
            "matrix": base_path / "argsme_tfidf_matrix_improved.joblib",
            "mapping": base_path / "argsme_doc_mapping_improved.joblib"
        }
    elif dataset_name == "wikir_en1k_training":
        return {
            "vectorizer": base_path / "wikir_en1k_training_tfidf_vectorizer.joblib",
            "matrix": base_path / "wikir_en1k_training_tfidf_matrix.joblib",
            "mapping": base_path / "wikir_en1k_training_doc_mapping.joblib"
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_embedding_paths(dataset_name: str) -> Dict[str, Path]:
    """
    Get embedding file paths for a specific dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing paths to embedding files
    """
    base_path = Path("data/vectors") / dataset_name / "embedding"
    
    if dataset_name == "wikir":
        return {
            "embeddings": base_path / "wikir_bert_embeddings.npy",
            "mapping": base_path / "wikir_bert_doc_mapping.joblib",
            "info": base_path / "wikir_bert_info.joblib"
        }
    elif dataset_name == "argsme":
        return {
            "embeddings": base_path / "argsme_bert_embeddings.npy",
            "mapping": base_path / "argsme_bert_doc_mapping.joblib",
            "info": base_path / "argsme_bert_info.joblib"
        }
    elif dataset_name == "wikir_en1k_training":
        return {
            "embeddings": base_path / "wikir_en1k_training_bert_embeddings.npy",
            "mapping": base_path / "wikir_en1k_training_bert_doc_mapping.joblib",
            "info": base_path / "wikir_en1k_training_bert_info.joblib"
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_inverted_index_paths(dataset_name: str) -> Dict[str, Path]:
    """
    Get inverted index file paths for a specific dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing paths to inverted index files
    """
    base_path = Path("data/vectors") / dataset_name
    
    return {
        "index": base_path / "inverted_index.joblib",
        "doc_terms": base_path / "doc_terms.joblib",
        "term_stats": base_path / "term_stats.joblib"
    }

def ensure_directory_exists(path: Path) -> None:
    """
    Ensure that a directory exists, create it if it doesn't
    
    Args:
        path: Path to the directory
    """
    path.mkdir(parents=True, exist_ok=True) 