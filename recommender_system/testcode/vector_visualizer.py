"""
Vector Database 3D Visualizer
Visualizes FAISS vector database as an interactive 3D scatter plot
"""

import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
import sys
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import threading

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from path_utils import (
    get_faiss_index_dir,
    get_product_images_dir,
    find_file,
    get_project_root
)


class VectorVisualizer:
    """3D visualization of vector database"""
    
    def __init__(self):
        self.vectors = None
        self.vectors_3d = None
        self.product_ids = []
        self.product_metadata = None
        self.dimension_reducer = None
        self.reduction_method = 'PCA'  # 'PCA' or 'TSNE'
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load vectors and metadata"""
        print("Loading vector database...")
        
        # Try to load from .npy file first (faster)
        npy_file = find_file("resnet50_features_pca512.npy")
        if npy_file and npy_file.exists():
            print(f"Loading vectors from: {npy_file}")
            self.vectors = np.load(str(npy_file)).astype('float32')
            print(f"Loaded {len(self.vectors)} vectors")
        else:
            # Load from FAISS index
            faiss_index_dir = get_faiss_index_dir()
            index_path = faiss_index_dir / "product_index.faiss"
            
            if not index_path.exists():
                raise FileNotFoundError(
                    f"Neither .npy file nor FAISS index found.\n"
                    f"Looking for:\n"
                    f"  - resnet50_features_pca512.npy\n"
                    f"  - {index_path}"
                )
            
            print(f"Loading vectors from FAISS index: {index_path}")
            index = faiss.read_index(str(index_path))
            ntotal = index.ntotal
            print(f"Reconstructing {ntotal} vectors from index...")
            
            # Reconstruct vectors
            dimension = index.d
            self.vectors = np.zeros((ntotal, dimension), dtype='float32')
            for i in range(ntotal):
                if (i + 1) % 1000 == 0:
                    print(f"  Progress: {i+1}/{ntotal}")
                self.vectors[i] = index.reconstruct(i)
        
        # Load product IDs
        faiss_index_dir = get_faiss_index_dir()
        ids_path = faiss_index_dir / "product_ids.pkl"
        if ids_path.exists():
            with open(ids_path, 'rb') as f:
                self.product_ids = pickle.load(f)
        else:
            # Try to load from metadata
            metadata_file = find_file("resnet50_metadata.csv")
            if metadata_file and metadata_file.exists():
                try:
                    self.product_metadata = pd.read_csv(
                        str(metadata_file), 
                        on_bad_lines='skip', 
                        encoding='utf-8'
                    )
                    if 'id' in self.product_metadata.columns:
                        self.product_ids = self.product_metadata['id'].astype(str).tolist()
                    else:
                        self.product_ids = [str(i) for i in range(len(self.vectors))]
                except:
                    self.product_ids = [str(i) for i in range(len(self.vectors))]
            else:
                self.product_ids = [str(i) for i in range(len(self.vectors))]
        
        # Load metadata if not already loaded
        if self.product_metadata is None:
            metadata_file = find_file("resnet50_metadata.csv")
            if metadata_file and metadata_file.exists():
                try:
                    self.product_metadata = pd.read_csv(
                        str(metadata_file),
                        on_bad_lines='skip',
                        encoding='utf-8'
                    )
                    print(f"Loaded metadata for {len(self.product_metadata)} products")
                except Exception as e:
                    print(f"Warning: Could not load metadata: {e}")
                    self.product_metadata = None
        
        print(f"✓ Loaded {len(self.vectors)} vectors")
        print(f"✓ Loaded {len(self.product_ids)} product IDs")
        if self.product_metadata is not None:
            print(f"✓ Loaded metadata for {len(self.product_metadata)} products")
    
    def reduce_dimensions(self, method='PCA', n_components=3, sample_size=None):
        """
        Reduce vector dimensions to 3D for visualization.
        
        Args:
            method: 'PCA' or 'TSNE'
            n_components: Number of dimensions (3 for 3D)
            sample_size: If not None, sample this many vectors for faster computation
        """
        print(f"\nReducing dimensions using {method}...")
        
        vectors_to_reduce = self.vectors
        
        # Sample if requested (for faster computation with large datasets)
        if sample_size and len(vectors_to_reduce) > sample_size:
            print(f"Sampling {sample_size} vectors from {len(vectors_to_reduce)} for faster computation...")
            indices = np.random.choice(len(vectors_to_reduce), sample_size, replace=False)
            vectors_to_reduce = vectors_to_reduce[indices]
            self.sample_indices = indices
        else:
            self.sample_indices = np.arange(len(vectors_to_reduce))
        
        if method == 'PCA':
            print("Computing PCA...")
            self.dimension_reducer = PCA(n_components=n_components, random_state=42)
            self.vectors_3d = self.dimension_reducer.fit_transform(vectors_to_reduce)
            print(f"✓ PCA explained variance: {self.dimension_reducer.explained_variance_ratio_.sum():.2%}")
        elif method == 'TSNE':
            print("Computing t-SNE (this may take a while)...")
            # Use PCA first for large datasets to avoid overflow issues
            if len(vectors_to_reduce) > 1000:
                print("  Pre-processing with PCA to 50 dimensions...")
                pca = PCA(n_components=50, random_state=42)
                vectors_pca = pca.fit_transform(vectors_to_reduce)
            else:
                vectors_pca = vectors_to_reduce
            
            # Calculate safe perplexity (must be less than n_samples)
            n_samples = len(vectors_pca)
            safe_perplexity = min(30, max(5, n_samples - 1))
            if safe_perplexity >= n_samples:
                safe_perplexity = max(5, n_samples // 4)
            
            print(f"  Using perplexity: {safe_perplexity} (n_samples: {n_samples})")
            
            # Use max_iter instead of n_iter (newer scikit-learn versions)
            self.dimension_reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=safe_perplexity,
                max_iter=1000,
                learning_rate='auto'  # Auto-adjust learning rate
            )
            self.vectors_3d = self.dimension_reducer.fit_transform(vectors_pca)
            print("✓ t-SNE complete")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.reduction_method = method
        print(f"✓ Reduced to {n_components}D: {self.vectors_3d.shape}")
    
    def visualize(self, color_by=None, sample_size=5000):
        """
        Create interactive 3D visualization.
        
        Args:
            color_by: Column name from metadata to color by (e.g., 'masterCategory', 'gender')
            sample_size: Maximum number of points to display (for performance)
        """
        # Reduce dimensions if not done
        if self.vectors_3d is None:
            # Use sampling for large datasets
            if len(self.vectors) > sample_size:
                self.reduce_dimensions(method='PCA', sample_size=sample_size)
            else:
                self.reduce_dimensions(method='PCA')
        
        # Prepare colors
        colors = 'blue'
        color_map = None
        
        if color_by and self.product_metadata is not None:
            # Get color mapping
            try:
                # Map to sample indices if sampling was used
                if hasattr(self, 'sample_indices'):
                    sampled_metadata = self.product_metadata.iloc[self.sample_indices]
                else:
                    sampled_metadata = self.product_metadata
                
                if color_by in sampled_metadata.columns:
                    unique_values = sampled_metadata[color_by].unique()
                    color_map = plt.cm.get_cmap('tab20', len(unique_values))
                    value_to_color = {val: color_map(i) for i, val in enumerate(unique_values)}
                    colors = [value_to_color.get(val, 'gray') for val in sampled_metadata[color_by]]
                    print(f"Coloring by {color_by}: {len(unique_values)} unique values")
            except Exception as e:
                print(f"Warning: Could not color by {color_by}: {e}")
                colors = 'blue'
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        scatter = ax.scatter(
            self.vectors_3d[:, 0],
            self.vectors_3d[:, 1],
            self.vectors_3d[:, 2],
            c=colors if isinstance(colors, (list, np.ndarray)) else colors,
            s=20,
            alpha=0.6,
            picker=True
        )
        
        # Labels
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title(f'Vector Database Visualization ({self.reduction_method})\n'
                    f'{len(self.vectors_3d)} vectors shown')
        
        # Add legend if colored
        if color_map and color_by:
            try:
                if hasattr(self, 'sample_indices'):
                    sampled_metadata = self.product_metadata.iloc[self.sample_indices]
                else:
                    sampled_metadata = self.product_metadata
                
                unique_values = sampled_metadata[color_by].unique()
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=value_to_color[val], 
                              markersize=10, label=str(val))
                    for val in unique_values[:20]  # Limit to 20 for readability
                ]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
            except:
                pass
        
        # Add click handler
        def on_pick(event):
            if event.artist != scatter:
                return
            
            ind = event.ind[0]
            
            # Get product ID
            if hasattr(self, 'sample_indices'):
                actual_idx = self.sample_indices[ind]
            else:
                actual_idx = ind
            
            product_id = self.product_ids[actual_idx] if actual_idx < len(self.product_ids) else str(actual_idx)
            
            # Get product info
            info = f"Product ID: {product_id}\nIndex: {actual_idx}"
            
            if self.product_metadata is not None:
                try:
                    product_row = self.product_metadata[
                        self.product_metadata['id'].astype(str) == str(product_id)
                    ]
                    if not product_row.empty:
                        product = product_row.iloc[0]
                        info = f"Product: {product.get('productDisplayName', product_id)}\n"
                        info += f"ID: {product_id}\n"
                        info += f"Type: {product.get('articleType', 'N/A')}\n"
                        info += f"Category: {product.get('masterCategory', 'N/A')}\n"
                        info += f"Color: {product.get('baseColour', 'N/A')}\n"
                        info += f"Gender: {product.get('gender', 'N/A')}"
                except:
                    pass
            
            # Show info in a message box
            root = tk.Tk()
            root.withdraw()  # Hide main window
            messagebox.showinfo("Product Information", info)
            root.destroy()
        
        fig.canvas.mpl_connect('pick_event', on_pick)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main entry point"""
    print("="*60)
    print("Vector Database 3D Visualizer")
    print("="*60)
    
    try:
        # Create visualizer
        visualizer = VectorVisualizer()
        
        # Ask user for preferences
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        # Ask for reduction method
        method = messagebox.askyesno(
            "Reduction Method",
            "Use t-SNE for better visualization?\n\n"
            "Yes = t-SNE (slower but better)\n"
            "No = PCA (faster)"
        )
        method = 'TSNE' if method else 'PCA'
        
        # Ask for sampling
        sample = messagebox.askyesno(
            "Sampling",
            f"Dataset has {len(visualizer.vectors)} vectors.\n\n"
            "Sample 5000 vectors for faster visualization?\n\n"
            "Yes = Sample 5000 (faster)\n"
            "No = Use all vectors (slower)"
        )
        sample_size = 5000 if sample and len(visualizer.vectors) > 5000 else None
        
        # Ask for coloring
        color_by = None
        if visualizer.product_metadata is not None:
            color_options = ['None', 'masterCategory', 'gender', 'articleType', 'baseColour']
            choice = tk.simpledialog.askstring(
                "Color By",
                "Color points by:\n\n"
                "Options: None, masterCategory, gender, articleType, baseColour\n\n"
                "Enter option (or leave blank for None):"
            )
            if choice and choice.lower() != 'none' and choice in color_options:
                color_by = choice
        
        root.destroy()
        
        # Reduce dimensions
        visualizer.reduce_dimensions(method=method, sample_size=sample_size)
        
        # Visualize
        print("\nOpening 3D visualization...")
        print("Click on points to see product information!")
        visualizer.visualize(color_by=color_by, sample_size=sample_size)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        # Show error in GUI
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"Failed to visualize vectors:\n\n{e}")
        root.destroy()


if __name__ == "__main__":
    main()

