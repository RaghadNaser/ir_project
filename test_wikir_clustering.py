import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.sparse import load_npz
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import TruncatedSVD

class WikiRClusteringTest:
    def __init__(self):
        self.results = {}
        
    def load_wikir_data(self):
        """Load WikiR TF-IDF and embedding data"""
        print("Loading WikiR data...")
        
        # Load TF-IDF data
        try:
            self.tfidf_matrix = load_npz('data/vectors/wikir/tfidf/wikir_tfidf_matrix.npz')
            self.tfidf_vectorizer = joblib.load('data/vectors/wikir/tfidf/wikir_tfidf_vectorizer.joblib')
            print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        except Exception as e:
            print(f"Error loading TF-IDF data: {e}")
            self.tfidf_matrix = None
            
        # Load embedding data
        try:
            self.embeddings = np.load('data/vectors/wikir/embedding/wikir_bert_embeddings.npy')
            print(f"Embeddings shape: {self.embeddings.shape}")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            self.embeddings = None
            
        # Load document mapping
        try:
            self.doc_mapping = pd.read_csv('data/vectors/wikir/tfidf/wikir_doc_mapping_fixed.tsv', sep='\t')
            print(f"Document mapping shape: {self.doc_mapping.shape}")
        except Exception as e:
            print(f"Error loading document mapping: {e}")
            self.doc_mapping = None
    
    def evaluate_clustering(self, data, labels, method_name, data_type):
        """Evaluate clustering quality using multiple metrics"""
        if len(np.unique(labels)) < 2:
            return None
            
        metrics = {}
        
        # Silhouette Score (higher is better, range: -1 to 1)
        try:
            metrics['silhouette'] = silhouette_score(data, labels)
        except:
            metrics['silhouette'] = None
            
        # Calinski-Harabasz Score (higher is better)
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(data, labels)
        except:
            metrics['calinski_harabasz'] = None
            
        # Davies-Bouldin Score (lower is better)
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(data, labels)
        except:
            metrics['davies_bouldin'] = None
            
        # Number of clusters
        metrics['n_clusters'] = len(np.unique(labels))
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique, counts))
        metrics['min_cluster_size'] = min(counts)
        metrics['max_cluster_size'] = max(counts)
        metrics['avg_cluster_size'] = np.mean(counts)
        
        return metrics
    
    def test_kmeans(self, data, data_type, max_clusters=10):
        """Test KMeans clustering with different numbers of clusters"""
        print(f"\nTesting KMeans on {data_type}...")
        
        # Standardize data for better clustering
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        results = {}
        
        for n_clusters in range(2, max_clusters + 1):
            print(f"  Testing with {n_clusters} clusters...")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data_scaled)
            
            metrics = self.evaluate_clustering(data_scaled, labels, 'KMeans', data_type)
            if metrics:
                results[n_clusters] = {
                    'labels': labels,
                    'metrics': metrics,
                    'inertia': kmeans.inertia_
                }
        
        return results
    
    def test_dbscan(self, data, data_type):
        """Test DBSCAN clustering with different parameters"""
        print(f"\nTesting DBSCAN on {data_type}...")
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        results = {}
        
        # Test different eps values
        eps_values = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        min_samples_values = [5, 10, 15, 20]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                print(f"  Testing DBSCAN with eps={eps}, min_samples={min_samples}...")
                
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(data_scaled)
                
                # Skip if all points are noise (-1) or only one cluster
                if len(np.unique(labels)) < 2 or np.all(labels == -1):
                    continue
                
                metrics = self.evaluate_clustering(data_scaled, labels, 'DBSCAN', data_type)
                if metrics:
                    key = f"eps_{eps}_min_samples_{min_samples}"
                    results[key] = {
                        'labels': labels,
                        'metrics': metrics,
                        'eps': eps,
                        'min_samples': min_samples
                    }
        
        return results
    
    def test_hierarchical(self, data, data_type, max_clusters=10):
        """Test Hierarchical clustering"""
        print(f"\nTesting Hierarchical clustering on {data_type}...")
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        results = {}
        
        for n_clusters in range(2, max_clusters + 1):
            print(f"  Testing with {n_clusters} clusters...")
            
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            labels = hierarchical.fit_predict(data_scaled)
            
            metrics = self.evaluate_clustering(data_scaled, labels, 'Hierarchical', data_type)
            if metrics:
                results[n_clusters] = {
                    'labels': labels,
                    'metrics': metrics
                }
        
        return results
    
    def visualize_clusters(self, data, labels, title, data_type, method_name):
        """Visualize clusters using PCA and t-SNE"""
        print(f"Visualizing {method_name} clusters for {data_type}...")
        
        # Reduce dimensions for visualization
        if data.shape[1] > 2:
            # Use PCA first for faster computation
            pca = PCA(n_components=min(50, data.shape[1]))
            data_pca = pca.fit_transform(data)
            
            # Then use t-SNE for better visualization
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data)//4))
            data_2d = tsne.fit_transform(data_pca)
        else:
            data_2d = data
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Scatter plot
        scatter = ax1.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
        ax1.set_title(f'{method_name} Clusters - {data_type}')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=ax1, label='Cluster')
        
        # Plot 2: Cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        ax2.bar(unique, counts, alpha=0.7)
        ax2.set_title('Cluster Size Distribution')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Documents')
        ax2.set_xticks(unique)
        
        plt.tight_layout()
        plt.savefig(f'wikir_clustering_{data_type}_{method_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_evaluation_metrics(self, results, data_type, method_name):
        """Plot evaluation metrics for different clustering configurations"""
        if not results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{method_name} Evaluation Metrics - {data_type}', fontsize=16)
        
        # Extract metrics
        n_clusters = []
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        
        for key, result in results.items():
            if isinstance(key, int):  # KMeans and Hierarchical
                n_clusters.append(key)
            else:  # DBSCAN
                n_clusters.append(result['metrics']['n_clusters'])
                
            silhouette_scores.append(result['metrics']['silhouette'])
            calinski_scores.append(result['metrics']['calinski_harabasz'])
            davies_scores.append(result['metrics']['davies_bouldin'])
        
        # Plot 1: Silhouette Score
        axes[0, 0].plot(n_clusters, silhouette_scores, 'bo-')
        axes[0, 0].set_title('Silhouette Score (Higher is Better)')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Calinski-Harabasz Score
        axes[0, 1].plot(n_clusters, calinski_scores, 'ro-')
        axes[0, 1].set_title('Calinski-Harabasz Score (Higher is Better)')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Calinski-Harabasz Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Davies-Bouldin Score
        axes[1, 0].plot(n_clusters, davies_scores, 'go-')
        axes[1, 0].set_title('Davies-Bouldin Score (Lower is Better)')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Davies-Bouldin Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Best configuration summary
        best_silhouette_idx = np.argmax(silhouette_scores)
        best_config = n_clusters[best_silhouette_idx]
        
        axes[1, 1].text(0.1, 0.8, f'Best Configuration:', fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, 0.7, f'Clusters: {best_config}', fontsize=10)
        axes[1, 1].text(0.1, 0.6, f'Silhouette: {silhouette_scores[best_silhouette_idx]:.3f}', fontsize=10)
        axes[1, 1].text(0.1, 0.5, f'Calinski: {calinski_scores[best_silhouette_idx]:.1f}', fontsize=10)
        axes[1, 1].text(0.1, 0.4, f'Davies: {davies_scores[best_silhouette_idx]:.3f}', fontsize=10)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'wikir_evaluation_{data_type}_{method_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comprehensive_test(self):
        """Run comprehensive clustering test on WikiR data"""
        print("=== WikiR Clustering Analysis ===")
        
        # Load data
        self.load_wikir_data()
        
        # Test TF-IDF data
        if self.tfidf_matrix is not None:
            print("\n" + "="*50)
            print("TESTING TF-IDF REPRESENTATION")
            print("="*50)
            
            sample_size = 500  # أو أقل حسب الذاكرة
            if self.tfidf_matrix.shape[0] > sample_size:
                indices = np.random.choice(self.tfidf_matrix.shape[0], sample_size, replace=False)
                tfidf_sample = self.tfidf_matrix[indices]
            else:
                tfidf_sample = self.tfidf_matrix

            # تقليل الأبعاد إلى 100 فقط مثلاً
            svd = TruncatedSVD(n_components=100, random_state=42)
            tfidf_reduced = svd.fit_transform(tfidf_sample)
            tfidf_dense = tfidf_reduced
            
            # Test different clustering methods
            tfidf_kmeans = self.test_kmeans(tfidf_dense, "TF-IDF")
            tfidf_dbscan = self.test_dbscan(tfidf_dense, "TF-IDF")
            tfidf_hierarchical = self.test_hierarchical(tfidf_dense, "TF-IDF")
            
            # Visualize best results
            if tfidf_kmeans:
                best_k = max(tfidf_kmeans.keys(), key=lambda k: tfidf_kmeans[k]['metrics']['silhouette'])
                self.visualize_clusters(tfidf_dense, tfidf_kmeans[best_k]['labels'], 
                                      f"KMeans (k={best_k})", "TF-IDF", "KMeans")
                self.plot_evaluation_metrics(tfidf_kmeans, "TF-IDF", "KMeans")
            
            self.results['tfidf'] = {
                'kmeans': tfidf_kmeans,
                'dbscan': tfidf_dbscan,
                'hierarchical': tfidf_hierarchical
            }
        
        # Test Embedding data
        if self.embeddings is not None:
            print("\n" + "="*50)
            print("TESTING EMBEDDING REPRESENTATION")
            print("="*50)
            
            # Test different clustering methods
            emb_kmeans = self.test_kmeans(self.embeddings, "Embeddings")
            emb_dbscan = self.test_dbscan(self.embeddings, "Embeddings")
            emb_hierarchical = self.test_hierarchical(self.embeddings, "Embeddings")
            
            # Visualize best results
            if emb_kmeans:
                best_k = max(emb_kmeans.keys(), key=lambda k: emb_kmeans[k]['metrics']['silhouette'])
                self.visualize_clusters(self.embeddings, emb_kmeans[best_k]['labels'], 
                                      f"KMeans (k={best_k})", "Embeddings", "KMeans")
                self.plot_evaluation_metrics(emb_kmeans, "Embeddings", "KMeans")
            
            self.results['embeddings'] = {
                'kmeans': emb_kmeans,
                'dbscan': emb_dbscan,
                'hierarchical': emb_hierarchical
            }
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print comprehensive summary of clustering results"""
        print("\n" + "="*60)
        print("CLUSTERING ANALYSIS SUMMARY")
        print("="*60)
        
        for data_type, methods in self.results.items():
            print(f"\n{data_type.upper()} REPRESENTATION:")
            print("-" * 40)
            
            for method_name, results in methods.items():
                if not results:
                    print(f"  {method_name}: No valid clusters found")
                    continue
                
                # Find best configuration
                if method_name == 'kmeans' or method_name == 'hierarchical':
                    best_key = max(results.keys(), key=lambda k: results[k]['metrics']['silhouette'])
                    best_result = results[best_key]
                    print(f"  {method_name.upper()} (Best k={best_key}):")
                else:  # DBSCAN
                    best_key = max(results.keys(), key=lambda k: results[k]['metrics']['silhouette'])
                    best_result = results[best_key]
                    print(f"  {method_name.upper()} (Best eps={best_result['eps']}, min_samples={best_result['min_samples']}):")
                
                metrics = best_result['metrics']
                print(f"    - Silhouette Score: {metrics['silhouette']:.3f}")
                print(f"    - Calinski-Harabasz: {metrics['calinski_harabasz']:.1f}")
                print(f"    - Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
                print(f"    - Number of clusters: {metrics['n_clusters']}")
                print(f"    - Cluster sizes: {metrics['min_cluster_size']} to {metrics['max_cluster_size']} (avg: {metrics['avg_cluster_size']:.1f})")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        best_overall = None
        best_score = -1
        
        for data_type, methods in self.results.items():
            for method_name, results in methods.items():
                if results:
                    if method_name == 'kmeans' or method_name == 'hierarchical':
                        best_key = max(results.keys(), key=lambda k: results[k]['metrics']['silhouette'])
                        score = results[best_key]['metrics']['silhouette']
                    else:
                        best_key = max(results.keys(), key=lambda k: results[k]['metrics']['silhouette'])
                        score = results[best_key]['metrics']['silhouette']
                    
                    if score > best_score:
                        best_score = score
                        best_overall = (data_type, method_name, best_key, results[best_key])
        
        if best_overall:
            data_type, method, config, result = best_overall
            print(f"Best overall clustering: {method.upper()} on {data_type}")
            print(f"Configuration: {config}")
            print(f"Silhouette Score: {result['metrics']['silhouette']:.3f}")
            
            if result['metrics']['silhouette'] > 0.3:
                print("✅ Clustering shows good structure - Consider using clustering in your pipeline")
            elif result['metrics']['silhouette'] > 0.1:
                print("⚠️  Clustering shows some structure - May be beneficial with careful tuning")
            else:
                print("❌ Clustering shows poor structure - May not be beneficial for this dataset")
        else:
            print("❌ No valid clustering configurations found")

if __name__ == "__main__":
    # Run the clustering test
    tester = WikiRClusteringTest()
    tester.run_comprehensive_test() 