"""
Shared utilities for evaluation notebooks
Common functions for loading data, generating test cases, and computing metrics
"""

import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add parent directory to path (recommender_system)
sys.path.insert(0, str(Path(__file__).parent.parent))
# Also add project root for finding files
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from path_utils import get_faiss_index_dir, find_file
from recommender import Recommender
from feature_extractor import FeatureExtractor


def load_vectors_and_metadata():
    """
    Load all vectors and metadata from vector DB.
    
    Returns:
        Tuple of (vectors, metadata, product_ids)
    """
    print("Loading vectors and metadata...")
    
    # Load vectors
    npy_file = find_file("resnet50_features_pca512.npy")
    if npy_file and npy_file.exists():
        print(f"Loading vectors from: {npy_file}")
        vectors = np.load(str(npy_file)).astype('float32')
    else:
        # Load from FAISS index
        faiss_index_dir = get_faiss_index_dir()
        index_path = faiss_index_dir / "product_index.faiss"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Vector file not found: {index_path}")
        
        print(f"Loading from FAISS index: {index_path}")
        index = faiss.read_index(str(index_path))
        ntotal = index.ntotal
        dimension = index.d
        
        print(f"Reconstructing {ntotal} vectors...")
        vectors = np.zeros((ntotal, dimension), dtype='float32')
        
        for i in range(ntotal):
            if (i + 1) % 5000 == 0:
                print(f"  Progress: {i+1}/{ntotal}")
            vectors[i] = index.reconstruct(i)
    
    print(f"Loaded {len(vectors)} vectors of dimension {vectors.shape[1]}")
    
    # Load metadata
    metadata_file = find_file("resnet50_metadata.csv")
    if not metadata_file or not metadata_file.exists():
        raise FileNotFoundError("Metadata file not found: resnet50_metadata.csv")
    
    print(f"Loading metadata from: {metadata_file}")
    try:
        metadata = pd.read_csv(str(metadata_file), on_bad_lines='skip', encoding='utf-8')
    except TypeError:
        try:
            metadata = pd.read_csv(str(metadata_file), error_bad_lines=False, warn_bad_lines=True, encoding='utf-8')
        except TypeError:
            metadata = pd.read_csv(str(metadata_file), error_bad_lines=False, encoding='utf-8')
    
    # Load product IDs
    faiss_index_dir = get_faiss_index_dir()
    ids_path = faiss_index_dir / "product_ids.pkl"
    if ids_path.exists():
        with open(ids_path, 'rb') as f:
            product_ids = pickle.load(f)
    else:
        # Use IDs from metadata
        product_ids = metadata['id'].astype(str).tolist()
    
    # Ensure alignment
    if len(vectors) != len(metadata):
        print(f"Warning: Vector count ({len(vectors)}) != metadata count ({len(metadata)})")
        min_len = min(len(vectors), len(metadata))
        vectors = vectors[:min_len]
        metadata = metadata.iloc[:min_len].reset_index(drop=True)
        product_ids = product_ids[:min_len]
    
    print(f"Loaded {len(metadata)} metadata records")
    return vectors, metadata, product_ids


def initialize_recommender():
    """Initialize the recommender system."""
    print("Initializing recommender...")
    recommender = Recommender()
    print("✓ Recommender initialized")
    return recommender


def generate_test_cases_by_color(metadata: pd.DataFrame, n_per_color: int = 10) -> Dict[str, List[int]]:
    """
    Generate test cases grouped by color.
    
    Args:
        metadata: Product metadata DataFrame
        n_per_color: Number of items per color
        
    Returns:
        Dictionary mapping color to list of product indices
    """
    test_cases = {}
    color_counts = metadata['baseColour'].value_counts()
    
    for color in color_counts.index:
        color_items = metadata[metadata['baseColour'] == color]
        if len(color_items) >= n_per_color:
            # Sample n_per_color items
            sampled = color_items.sample(n=n_per_color, random_state=42)
            test_cases[color] = sampled.index.tolist()
    
    return test_cases


def generate_test_cases_by_type(metadata: pd.DataFrame, n_per_type: int = 10) -> Dict[str, List[int]]:
    """
    Generate test cases grouped by articleType.
    
    Args:
        metadata: Product metadata DataFrame
        n_per_type: Number of items per type
        
    Returns:
        Dictionary mapping articleType to list of product indices
    """
    test_cases = {}
    type_counts = metadata['articleType'].value_counts()
    
    for article_type in type_counts.index:
        type_items = metadata[metadata['articleType'] == article_type]
        if len(type_items) >= n_per_type:
            sampled = type_items.sample(n=min(n_per_type, len(type_items)), random_state=42)
            test_cases[article_type] = sampled.index.tolist()
    
    return test_cases


def extract_pattern_type(product_name: str) -> str:
    """
    Extract pattern type from product name.
    
    Args:
        product_name: Product display name
        
    Returns:
        Pattern type: 'Solid', 'Striped', 'Printed', 'Patterned', or 'Unknown'
    """
    name_lower = str(product_name).lower()
    
    if any(word in name_lower for word in ['striped', 'stripe', 'strip']):
        return 'Striped'
    elif any(word in name_lower for word in ['printed', 'print', 'pattern']):
        return 'Printed'
    elif any(word in name_lower for word in ['solid', 'plain']):
        return 'Solid'
    elif any(word in name_lower for word in ['pattern', 'design', 'floral', 'geometric']):
        return 'Patterned'
    else:
        return 'Unknown'


def get_recommendations_for_items(
    recommender: Recommender,
    metadata: pd.DataFrame,
    item_indices: List[int],
    top_k: int = 20,
    strategy: str = 'hybrid'
) -> pd.DataFrame:
    """
    Get recommendations for a set of items.
    
    Args:
        recommender: Recommender instance
        metadata: Product metadata DataFrame
        item_indices: List of product indices to use as queries
        top_k: Number of recommendations
        strategy: Recommendation strategy
        
    Returns:
        DataFrame with recommendations
    """
    # Note: This is a simplified version - in practice, you'd need actual image paths
    # For evaluation, we'll use the feature vectors directly
    # This function would need to be adapted based on how you want to test
    
    # For now, return empty - this will be implemented per notebook
    return pd.DataFrame()


def compute_color_match_rate(recommendations: pd.DataFrame, query_color: str) -> float:
    """
    Compute the percentage of recommendations that match the query color.
    
    Args:
        recommendations: DataFrame with recommendations
        query_color: Query color
        
    Returns:
        Color match rate (0-1)
    """
    if len(recommendations) == 0:
        return 0.0
    
    matches = (recommendations['baseColour'] == query_color).sum()
    return matches / len(recommendations)


def compute_diversity_score(series: pd.Series) -> float:
    """
    Compute diversity score (unique values / total).
    
    Args:
        series: Series to compute diversity for
        
    Returns:
        Diversity score (0-1)
    """
    if len(series) == 0:
        return 0.0
    
    unique_count = series.nunique()
    return unique_count / len(series)


def compute_cluster_metrics(vectors: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute clustering quality metrics.
    
    Args:
        vectors: Feature vectors
        labels: Cluster labels
        
    Returns:
        Dictionary with metrics
    """
    if len(np.unique(labels)) < 2:
        return {
            'silhouette_score': -1.0,
            'davies_bouldin_score': float('inf'),
            'n_clusters': len(np.unique(labels))
        }
    
    # Sample if too large for silhouette score
    if len(vectors) > 10000:
        indices = np.random.choice(len(vectors), 10000, replace=False)
        sample_vectors = vectors[indices]
        sample_labels = labels[indices]
    else:
        sample_vectors = vectors
        sample_labels = labels
    
    try:
        silhouette = silhouette_score(sample_vectors, sample_labels)
    except:
        silhouette = -1.0
    
    try:
        db_score = davies_bouldin_score(vectors, labels)
    except:
        db_score = float('inf')
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_score': db_score,
        'n_clusters': len(np.unique(labels))
    }


def compute_cluster_purity(labels: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Compute cluster purity (homogeneity).
    
    Args:
        labels: Cluster labels
        true_labels: True category labels
        
    Returns:
        Purity score (0-1)
    """
    if len(labels) != len(true_labels):
        return 0.0
    
    purity = 0.0
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_true_labels = true_labels[cluster_mask]
        
        if len(cluster_true_labels) > 0:
            most_common = pd.Series(cluster_true_labels).mode()[0]
            purity += (cluster_true_labels == most_common).sum()
    
    return purity / len(labels)


def reduce_dimensions(vectors: np.ndarray, n_components: int = 3, method: str = 'PCA', sample_size: Optional[int] = None) -> np.ndarray:
    """
    Reduce dimensions for visualization.
    
    Args:
        vectors: High-dimensional vectors
        n_components: Target dimensions
        method: 'PCA' or 'TSNE'
        sample_size: Optional sampling for large datasets
        
    Returns:
        Reduced dimension vectors
    """
    if sample_size and len(vectors) > sample_size:
        indices = np.random.choice(len(vectors), sample_size, replace=False)
        vectors = vectors[indices]
    
    if method == 'PCA':
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(vectors)
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        return reduced
    elif method == 'TSNE':
        tsne = TSNE(n_components=n_components, random_state=42, verbose=1)
        reduced = tsne.fit_transform(vectors)
        return reduced
    else:
        raise ValueError(f"Unknown method: {method}")


def save_plot(fig, filename: str, subdir: str = ""):
    """
    Save plot to evaluation_results/plots directory.
    
    Args:
        fig: Matplotlib figure
        filename: Filename
        subdir: Optional subdirectory
    """
    plots_dir = Path(__file__).parent.parent / "evaluation_results" / "plots"
    if subdir:
        plots_dir = plots_dir / subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = plots_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filepath}")


def save_metrics(metrics: Dict, filename: str):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filename: Filename
    """
    import json
    metrics_dir = Path(__file__).parent.parent / "evaluation_results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = metrics_dir / filename
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {filepath}")

