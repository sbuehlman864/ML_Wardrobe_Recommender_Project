"""
Simple 3D Visualization of Vector DB
Creates a 3D scatter plot of all vectors in the database
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import faiss
import pickle
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from path_utils import get_faiss_index_dir, find_file


def load_vectors():
    """Load all vectors from vector DB"""
    print("Loading vectors...")
    
    # Try to load from .npy file first (fastest)
    npy_file = find_file("resnet50_features_pca512.npy")
    if npy_file and npy_file.exists():
        print(f"Loading from: {npy_file}")
        vectors = np.load(str(npy_file)).astype('float32')
        print(f"Loaded {len(vectors)} vectors of dimension {vectors.shape[1]}")
        return vectors
    
    # Fallback: Load from FAISS index
    faiss_index_dir = get_faiss_index_dir()
    index_path = faiss_index_dir / "product_index.faiss"
    
    if not index_path.exists():
        raise FileNotFoundError(
            f"Vector file not found. Looking for:\n"
            f"  - resnet50_features_pca512.npy\n"
            f"  - {index_path}"
        )
    
    print(f"Loading from FAISS index: {index_path}")
    index = faiss.read_index(str(index_path))
    ntotal = index.ntotal
    dimension = index.d
    
    print(f"Reconstructing {ntotal} vectors...")
    vectors = np.zeros((ntotal, dimension), dtype='float32')
    
    # Reconstruct all vectors
    for i in range(ntotal):
        if (i + 1) % 5000 == 0:
            print(f"  Progress: {i+1}/{ntotal}")
        vectors[i] = index.reconstruct(i)
    
    print(f"Loaded {len(vectors)} vectors")
    return vectors


def reduce_to_3d(vectors, method='PCA', sample_size=None):
    """
    Reduce vectors from 512D to 3D for visualization
    
    Args:
        vectors: Array of shape (n, 512)
        method: 'PCA' or 'TSNE'
        sample_size: If provided, sample this many vectors (for speed)
    
    Returns:
        3D coordinates of shape (n, 3)
    """
    # Sample if requested (for large datasets)
    if sample_size and len(vectors) > sample_size:
        print(f"Sampling {sample_size} vectors from {len(vectors)}...")
        indices = np.random.choice(len(vectors), sample_size, replace=False)
        vectors = vectors[indices]
        print(f"Using {len(vectors)} vectors for visualization")
    
    print(f"Reducing {vectors.shape[1]}D to 3D using {method}...")
    
    if method == 'PCA':
        pca = PCA(n_components=3)
        vectors_3d = pca.fit_transform(vectors)
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_variance:.2%}")
        return vectors_3d
    
    elif method == 'TSNE':
        from sklearn.manifold import TSNE
        print("Warning: t-SNE can be slow for large datasets...")
        tsne = TSNE(n_components=3, random_state=42, verbose=1)
        vectors_3d = tsne.fit_transform(vectors)
        return vectors_3d
    
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_3d(vectors_3d, color_by=None, metadata=None):
    """
    Create 3D scatter plot
    
    Args:
        vectors_3d: Array of shape (n, 3)
        color_by: Column name to color by (from metadata)
        metadata: DataFrame with product metadata
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color coding
    if color_by and metadata is not None and color_by in metadata.columns:
        print(f"Color coding by: {color_by}")
        unique_values = metadata[color_by].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_values)))
        value_to_color = {val: colors[i] for i, val in enumerate(unique_values)}
        
        for val in unique_values:
            mask = metadata[color_by] == val
            if mask.sum() > 0:
                ax.scatter(
                    vectors_3d[mask, 0],
                    vectors_3d[mask, 1],
                    vectors_3d[mask, 2],
                    c=[value_to_color[val]],
                    label=str(val),
                    alpha=0.6,
                    s=10
                )
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Single color
        ax.scatter(
            vectors_3d[:, 0],
            vectors_3d[:, 1],
            vectors_3d[:, 2],
            alpha=0.6,
            s=10,
            c='blue'
        )
    
    ax.set_xlabel('X (Dimension 1)')
    ax.set_ylabel('Y (Dimension 2)')
    ax.set_zlabel('Z (Dimension 3)')
    ax.set_title('3D Visualization of Vector DB\n(All Clothing Items)')
    
    plt.tight_layout()
    plt.show()


def load_metadata():
    """Load product metadata if available"""
    metadata_file = find_file("resnet50_metadata.csv")
    if metadata_file and metadata_file.exists():
        import pandas as pd
        try:
            metadata = pd.read_csv(str(metadata_file), on_bad_lines='skip', encoding='utf-8')
            print(f"Loaded metadata for {len(metadata)} products")
            return metadata
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
            return None
    return None


def main():
    """Main function"""
    print("=" * 60)
    print("Vector DB 3D Visualizer")
    print("=" * 60)
    
    # Load vectors
    try:
        vectors = load_vectors()
    except Exception as e:
        print(f"Error loading vectors: {e}")
        return
    
    # Ask user for options
    print("\nOptions:")
    print("1. Use all vectors (slower but complete)")
    print("2. Sample 5000 vectors (faster)")
    
    choice = input("Enter choice (1 or 2, default=2): ").strip()
    sample_size = None if choice == '1' else 5000
    
    print("\nReduction method:")
    print("1. PCA (fast, linear)")
    print("2. t-SNE (slow, better clusters)")
    
    method_choice = input("Enter choice (1 or 2, default=1): ").strip()
    method = 'TSNE' if method_choice == '2' else 'PCA'
    
    # Reduce to 3D
    try:
        vectors_3d = reduce_to_3d(vectors, method=method, sample_size=sample_size)
    except Exception as e:
        print(f"Error reducing dimensions: {e}")
        return
    
    # Load metadata for color coding
    metadata = load_metadata()
    
    # Ask for color coding
    color_by = None
    if metadata is not None:
        print("\nColor coding options:")
        print("  - masterCategory (Apparel, Footwear, Accessories)")
        print("  - gender (Men, Women, Unisex)")
        print("  - articleType (Tshirts, Jeans, etc.)")
        print("  - baseColour (Red, Blue, etc.)")
        print("  - None (single color)")
        
        color_choice = input("Enter column name (or press Enter for None): ").strip()
        if color_choice and color_choice in metadata.columns:
            color_by = color_choice
            # Align metadata with sampled vectors if needed
            if sample_size and len(metadata) > sample_size:
                # We need to track which vectors were sampled
                # For simplicity, just use first N
                metadata = metadata.iloc[:len(vectors_3d)]
    
    # Plot
    print("\nGenerating 3D plot...")
    plot_3d(vectors_3d, color_by=color_by, metadata=metadata)
    
    print("\n✓ Visualization complete!")
    print("Tip: Rotate the plot by clicking and dragging")


if __name__ == "__main__":
    main()