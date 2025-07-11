import joblib
import numpy as np
from collections import defaultdict, Counter
from typing import List, Set, Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from config.logging_config import setup_logging
from utils.path_utils import get_inverted_index_paths

logger = setup_logging("inverted_index")

class InvertedIndex:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.index = defaultdict(list)  # term -> [(doc_id, positions), ...]
        self.doc_terms = defaultdict(set)  # doc_id -> set of terms
        self.term_stats = defaultdict(dict)  # term -> {df, idf, etc.}
        self.loaded = False
    
    def build_index(self, force_rebuild: bool = False):
        """Build or load inverted index"""
        try:
            paths = get_inverted_index_paths(self.dataset_name)
            
            # Try to load existing index
            if not force_rebuild:
                try:
                    # Try to load the combined index file first
                    combined_index_path = paths['index'].parent / f"{self.dataset_name}_inverted_index.joblib"
                    if combined_index_path.exists():
                        data = joblib.load(combined_index_path)
                        self.index = data.get("index", {})
                        # Create doc_terms from index
                        self.doc_terms = defaultdict(set)
                        for term, doc_data in self.index.items():
                            for doc_id in doc_data.keys():
                                self.doc_terms[doc_id].add(term)
                        self.doc_terms = dict(self.doc_terms)
                        # Create term_stats from index
                        self.term_stats = {}
                        for term, doc_data in self.index.items():
                            self.term_stats[term] = {
                                "df": len(doc_data),
                                "total_tf": sum(doc_info.get("tf", 0) for doc_info in doc_data.values())
                            }
                        self.loaded = True
                        logger.info(f"Combined inverted index loaded for {self.dataset_name}")
                        return
                    else:
                        # Try to load separate files
                        self.index = joblib.load(paths['index'])
                        self.doc_terms = joblib.load(paths['doc_terms'])
                        self.term_stats = joblib.load(paths['term_stats'])
                        self.loaded = True
                        logger.info(f"Inverted index loaded for {self.dataset_name}")
                        return
                except FileNotFoundError:
                    logger.info(f"No existing index found for {self.dataset_name}, skipping inverted index")
                    self.loaded = False
                    return
            
            # Build new index (placeholder - would need actual document data)
            logger.info(f"Building inverted index for {self.dataset_name}")
            # This is a placeholder - in real implementation you'd process documents here
            
            self.loaded = True
            
        except Exception as e:
            logger.error(f"Error building inverted index: {e}")
            self.loaded = False
    
    def get_candidate_documents(self, query_terms: List[str], method: str = "union") -> Set[str]:
        """Get candidate documents for query terms"""
        if not self.loaded:
            return set()
        
        candidates = set()
        for term in query_terms:
            if term in self.index:
                doc_ids = [doc_id for doc_id, _ in self.index[term]]
                candidates.update(doc_ids)
        
        return candidates
    
    def search_with_ranking(self, query_terms: List[str], top_k: int = 10, ranking_method: str = "tfidf") -> List[Tuple[str, float]]:
        """Search with ranking"""
        if not self.loaded:
            return []
        
        # Simple implementation - return empty results for now
        return []
    
    def get_term_frequency(self, doc_id: str, term: str) -> int:
        """Get term frequency in document"""
        if not self.loaded or term not in self.index:
            return 0
        
        for d_id, positions in self.index[term]:
            if d_id == doc_id:
                return len(positions)
        return 0
    
    def get_term_positions(self, doc_id: str, term: str) -> List[int]:
        """Get term positions in document"""
        if not self.loaded or term not in self.index:
            return []
        
        for d_id, positions in self.index[term]:
            if d_id == doc_id:
                return positions
        return []
    
    def get_term_tfidf(self, doc_id: str, term: str) -> float:
        """Get TF-IDF score for term in document"""
        # Placeholder implementation
        return 0.0
    
    def get_index_statistics(self) -> Dict:
        """Get index statistics"""
        if not self.loaded:
            return {}
        
        return {
            'total_terms': len(self.index),
            'total_documents': len(self.doc_terms),
            'avg_terms_per_doc': np.mean([len(terms) for terms in self.doc_terms.values()]) if self.doc_terms else 0
        } 