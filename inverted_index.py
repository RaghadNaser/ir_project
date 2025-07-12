import joblib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
from tqdm import tqdm
from utils.database_utils import DatasetService
from config.settings import VECTORS_PATH
import warnings
import pickle

# Try to import PyTorch for GPU acceleration
try:
    import torch
    from sklearn.feature_extraction.text import CountVectorizer
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    import pandas as pd

class InvertedIndex:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.index = defaultdict(dict)  # {term: {doc_id: {tf, tfidf}}}
        self.doc_lengths = {}
        self.total_docs = 0
        self.dataset_path = VECTORS_PATH / dataset
        self.dataset_path.mkdir(exist_ok=True)
        self.batch_size = 5000  # Process documents in batches

    def load_cleaned_documents(self):
        """Load documents from the manually cleaned dataset"""
        if self.dataset == "argsme":
            cleaned_path = Path("data/vectors/argsme/processed/ARGSME_cleaned_docs.tsv")
            if not cleaned_path.exists():
                # Try alternative path structure
                cleaned_path = Path("data/processed/argsme/ARGSME_cleaned_docs.tsv")
            
            if not cleaned_path.exists():
                raise FileNotFoundError(f"Cleaned ARGSME documents not found. Expected at: {cleaned_path}")
                
            df = pd.read_csv(cleaned_path, sep='\t')
            docs_list = []
            
            for _, row in df.iterrows():
                docs_list.append({
                    'doc_id': str(row['doc_id']),
                    'combined_text': str(row['processed_text'])  # Use processed_text as combined_text
                })
            
            return docs_list
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")

    def build_index(self, force_rebuild: bool = False):
        """Build inverted index with TF and TF-IDF using GPU (PyTorch) if available"""
        index_file = self.dataset_path / 'inverted_index.joblib'
        
        if index_file.exists() and not force_rebuild:
            data = joblib.load(index_file)
            self.index = data['index']
            self.doc_lengths = data.get('doc_lengths', {})
            self.total_docs = data.get('total_docs', 0)
            return

        docs_list = self.load_cleaned_documents()
        self.total_docs = len(docs_list)
        
        # Extract necessary data before processing
        doc_ids = [str(doc['doc_id']) for doc in docs_list]
        raw_texts = [doc['combined_text'] for doc in docs_list]
        self.doc_lengths = {doc_id: len(text.split()) for doc_id, text in zip(doc_ids, raw_texts)}

        print(f"Building inverted index for {self.total_docs:,} documents...")

        if GPU_AVAILABLE:
            print("Using GPU acceleration with PyTorch for TF-IDF calculation.")
            device = torch.device("cuda")
            
            # Since text is already tokenized, use different tokenization approach
            print("Vectorizing pre-processed documents (CPU)...")
            count_vectorizer = CountVectorizer(
                token_pattern=r'\b\w+\b',  # Simple word boundaries
                lowercase=False,  # Text already lowercased
                preprocessor=None,  # No additional preprocessing
                tokenizer=None  # Use default tokenizer
            )
            tf_sparse_scipy = count_vectorizer.fit_transform(raw_texts)
            terms = count_vectorizer.get_feature_names_out()
            num_terms = len(terms)

            # 2. Move data to GPU with PyTorch
            print("Moving data to GPU and calculating TF-IDF...")
            tf_indices = torch.from_numpy(np.vstack(tf_sparse_scipy.nonzero())).to(device)
            tf_values = torch.from_numpy(tf_sparse_scipy.data).to(device)
            tf_shape = tf_sparse_scipy.shape
            
            tf_sparse_torch = torch.sparse_coo_tensor(tf_indices, tf_values, tf_shape, dtype=torch.float32)
            # Coalesce the tensor to sum up duplicate indices, which is required for further operations
            tf_sparse_torch = tf_sparse_torch.coalesce()

            # 3. Calculate IDF on GPU
            # Document frequency
            df = torch.bincount(tf_sparse_torch.indices()[1], minlength=num_terms)
            # Inverse document frequency
            idf = torch.log(self.total_docs / (df + 1.0))

            # 4. Calculate TF-IDF on GPU
            # Element-wise multiplication of TF with corresponding IDF
            tfidf_values = tf_sparse_torch.values() * idf[tf_sparse_torch.indices()[1]]
            
            # 5. Build the final index from GPU tensors
            print("Building inverted index from GPU tensors...")
            tf_indices_cpu = tf_indices.cpu().numpy()
            tf_values_cpu = tf_values.cpu().numpy()
            tfidf_values_cpu = tfidf_values.cpu().numpy()

            for i in tqdm(range(tf_indices_cpu.shape[1]), desc="Populating index"):
                doc_idx, term_idx = tf_indices_cpu[:, i]
                doc_id = doc_ids[doc_idx]
                term = terms[term_idx]
                
                self.index[term][doc_id] = {
                    "tf": tf_values_cpu[i].item(),
                    "tfidf": tfidf_values_cpu[i].item()
                }
        else:
            warnings.warn("PyTorch CUDA not available. Falling back to CPU-based implementation.")
            import pandas as pd
            df = pd.DataFrame(docs_list)

            from sklearn.feature_extraction.text import TfidfVectorizer
            # Configure vectorizer for pre-processed text
            vectorizer = TfidfVectorizer(
                token_pattern=r'\b\w+\b',  # Simple word boundaries
                lowercase=False,  # Text already lowercased
                preprocessor=None,  # No additional preprocessing
                tokenizer=None  # Use default tokenizer
            )
            tfidf_matrix = vectorizer.fit_transform(df['combined_text'].fillna(""))
            terms = vectorizer.get_feature_names_out()
            
            tf_vectorizer = CountVectorizer(
                vocabulary=terms,
                token_pattern=r'\b\w+\b',
                lowercase=False,
                preprocessor=None,
                tokenizer=None
            )
            tf_matrix = tf_vectorizer.fit_transform(df['combined_text'].fillna(""))

            print("Building inverted index from TF-IDF matrix (CPU)...")
            tfidf_csr = tfidf_matrix.tocsr()
            tf_csr = tf_matrix.tocsr()
            
            rows, cols = tfidf_csr.nonzero()
            for row, col in tqdm(zip(rows, cols), total=len(rows), desc="Populating index"):
                doc_id = doc_ids[row]
                term = terms[col]
                self.index[term][doc_id] = {
                    "tf": tf_csr[row, col],
                    "tfidf": tfidf_csr[row, col]
                }

        # Save index with a progress bar
        data_to_save = {
            'index': dict(self.index),
            'doc_lengths': self.doc_lengths,
            'total_docs': self.total_docs
        }

        # To show progress for saving, we first serialize the data to memory to get its size,
        # then write the serialized data to disk with a tqdm progress bar.
        pickled_data = pickle.dumps(data_to_save)
        
        with open(index_file, 'wb') as f:
            with tqdm.wrapattr(
                f, "write",
                total=len(pickled_data),
                desc=f"Saving index to {index_file}",
                unit='B', unit_scale=True, unit_divisor=1024
            ) as file_out:
                file_out.write(pickled_data)

        print("✅ Index build and save complete.")

    def get_candidate_documents(self, query_terms: List[str], method: str = "union") -> Set[str]:
        """Return candidate documents using union or intersection method"""
        if not query_terms:
            return set()

        if method == "union":
            candidate_docs = set()
            for term in query_terms:
                if term in self.index:
                    candidate_docs.update(self.index[term].keys())

        elif method == "intersection":
            candidate_docs = None
            for term in query_terms:
                if term in self.index:
                    if candidate_docs is None:
                        candidate_docs = set(self.index[term].keys())
                    else:
                        candidate_docs &= set(self.index[term].keys())
            candidate_docs = candidate_docs or set()

        else:
            raise ValueError(f"Unknown method: {method}")

        return candidate_docs

    def get_term_frequency(self, doc_id: str, term: str) -> int:
        """Return term frequency in a specific document"""
        if term in self.index and doc_id in self.index[term]:
            return self.index[term][doc_id]["tf"]
        return 0

    def get_term_positions(self, doc_id: str, term: str) -> List[int]:
        """Return positions of a term in a document"""
        if term in self.index and doc_id in self.index[term]:
            return self.index[term][doc_id]["positions"]
        return []

    def get_term_tfidf(self, doc_id: str, term: str) -> float:
        """Return TF-IDF score of a term in a document"""
        if term in self.index and doc_id in self.index[term]:
            return self.index[term][doc_id]["tfidf"]
        return 0.0

    def search_with_ranking(self, query_terms: List[str], top_k: int = 10, ranking_method: str = "tfidf") -> List[Tuple[str, float]]:
        """Search and return top_k documents ranked by tfidf/tf"""
        candidate_docs = self.get_candidate_documents(query_terms, method="union")
        if not candidate_docs:
            return []

        doc_scores = {}
        for doc_id in candidate_docs:
            score = 0.0
            for term in query_terms:
                if term in self.index and doc_id in self.index[term]:
                    if ranking_method == "tfidf":
                        score += self.index[term][doc_id]["tfidf"]
                    else:  # tf
                        score += self.index[term][doc_id]["tf"]

            if score > 0:
                doc_scores[doc_id] = score

        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def get_index_statistics(self) -> Dict:
        """Return index statistics"""
        total_terms = len(self.index)
        total_postings = sum(len(docs) for docs in self.index.values())
        avg_doc_length = np.mean(list(self.doc_lengths.values())) if self.doc_lengths else 0

        return {
            "total_terms": total_terms,
            "total_postings": total_postings,
            "total_documents": self.total_docs,
            "average_document_length": avg_doc_length,
            "index_size_mb": self._get_index_size()
        }

    def _get_index_size(self) -> float:
        """Estimate index size in MB"""
        import sys
        size = sys.getsizeof(self.index)
        for term, docs in self.index.items():
            size += sys.getsizeof(term)
            for doc_id, info in docs.items():
                size += sys.getsizeof(doc_id) + sys.getsizeof(info)
        return size / (1024 * 1024)
