"""
Vector Database Manager using FAISS
Manages FAISS indices for product features and user wardrobe features
"""

import numpy as np
import pickle
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import faiss
from path_utils import (
    get_faiss_index_dir,
    get_user_features_path,
    find_file,
    get_project_root
)


class VectorDB:
    """Manage FAISS indices for efficient similarity search"""
    
    def __init__(self, 
                 dimension: int = 512,
                 index_type: str = 'flat',
                 auto_init: bool = True):
        """
        Initialize vector database.
        
        Args:
            dimension: Feature vector dimension (default: 512)
            index_type: Type of FAISS index ('flat' for exact search)
            auto_init: If True, automatically check and create index if needed
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.product_ids = []  # Mapping from index position to product ID
        self.index_dir = get_faiss_index_dir()
        self.index_path = self.index_dir / "product_index.faiss"
        self.ids_path = self.index_dir / "product_ids.pkl"
        
        if auto_init:
            self._ensure_index_exists()
    
    def _ensure_index_exists(self) -> bool:
        """
        Check if FAISS index exists, create if not from .npy file.
        
        Returns:
            True if index is ready, False otherwise
        """
        # Check if index already exists
        if self.index_path.exists() and self.ids_path.exists():
            print("Loading existing FAISS index...")
            self.load_index()
            print(f"✓ Loaded FAISS index with {len(self.product_ids)} products")
            return True
        
        # Index doesn't exist, try to create from .npy file
        print("FAISS index not found. Checking for .npy file...")
        npy_file = find_file("resnet50_features_pca512.npy")
        metadata_file = find_file("resnet50_metadata.csv")
        
        if npy_file and npy_file.exists():
            if metadata_file and metadata_file.exists():
                print("Creating FAISS index from .npy file...")
                self._create_index_from_npy(str(npy_file), str(metadata_file))
                print(f"✓ FAISS index created successfully with {len(self.product_ids)} products")
                return True
            else:
                print("Warning: .npy file found but metadata CSV not found")
                return False
        else:
            raise FileNotFoundError(
                "Neither FAISS index nor .npy file found.\n"
                "Please ensure one of the following exists:\n"
                f"1. FAISS index: {self.index_path}\n"
                "2. Feature file: resnet50_features_pca512.npy\n\n"
                "If .npy file exists, the index will be created automatically on first run."
            )
    
    def _create_index_from_npy(self, npy_path: str, metadata_path: str):
        """
        Create FAISS index from existing .npy file (one-time migration).
        
        Args:
            npy_path: Path to resnet50_features_pca512.npy
            metadata_path: Path to resnet50_metadata.csv
        """
        import pandas as pd
        
        # Load features
        print(f"Loading features from {npy_path}...")
        features = np.load(npy_path).astype('float32')
        print(f"Loaded {len(features)} feature vectors")
        
        # Load metadata to get product IDs
        print(f"Loading metadata from {metadata_path}...")
        try:
            metadata = pd.read_csv(metadata_path, on_bad_lines='skip', encoding='utf-8')
        except TypeError:
            try:
                metadata = pd.read_csv(metadata_path, error_bad_lines=False, warn_bad_lines=True, encoding='utf-8')
            except TypeError:
                metadata = pd.read_csv(metadata_path, error_bad_lines=False, encoding='utf-8')
        
        # Get product IDs
        if 'id' in metadata.columns:
            product_ids = metadata['id'].astype(str).tolist()
        else:
            # Fallback: use index as ID
            product_ids = [str(i) for i in range(len(features))]
        
        # Normalize vectors for cosine similarity (L2 normalization)
        print("Normalizing vectors...")
        faiss.normalize_L2(features)
        
        # Create FAISS index
        print("Creating FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
        
        # Add vectors to index
        print("Adding vectors to index...")
        self.index.add(features)
        self.product_ids = product_ids
        
        # Save index
        print("Saving FAISS index...")
        self.save_index()
        
        print("✓ Index creation complete")
    
    def add_products(self, features: np.ndarray, ids: List[str]):
        """
        Add product features to the index.
        
        Args:
            features: Feature vectors (N x dimension)
            ids: Product IDs corresponding to features
        """
        if self.index is None:
            # Create new index if it doesn't exist
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Ensure features are float32
        features = features.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(features)
        
        # Add to index
        self.index.add(features)
        self.product_ids.extend(ids)
        
        # Save updated index
        self.save_index()
    
    def add_user_features(self, 
                         user_id: str, 
                         features: np.ndarray, 
                         image_paths: List[str]):
        """
        Store user wardrobe features.
        
        Args:
            user_id: User identifier
            features: Feature vectors (N x dimension)
            image_paths: List of image paths corresponding to features
        """
        user_dir = get_user_features_path(user_id)
        
        # Save features
        features_path = user_dir / f"{user_id}_features.npy"
        np.save(str(features_path), features.astype('float32'))
        
        # Save metadata
        metadata = {
            'user_id': user_id,
            'image_paths': image_paths,
            'num_features': len(features),
            'feature_dim': features.shape[1] if len(features.shape) > 1 else features.shape[0]
        }
        metadata_path = user_dir / f"{user_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_user_features(self, user_id: str) -> Optional[np.ndarray]:
        """
        Retrieve user wardrobe features.
        
        Args:
            user_id: User identifier
        
        Returns:
            Feature vectors if found, None otherwise
        """
        user_dir = get_user_features_path(user_id)
        features_path = user_dir / f"{user_id}_features.npy"
        
        if features_path.exists():
            return np.load(str(features_path))
        return None
    
    def get_user_metadata(self, user_id: str) -> Optional[Dict]:
        """
        Retrieve user wardrobe metadata.
        
        Args:
            user_id: User identifier
        
        Returns:
            Metadata dict if found, None otherwise
        """
        user_dir = get_user_features_path(user_id)
        metadata_path = user_dir / f"{user_id}_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def search(self, 
              query_features: np.ndarray, 
              top_k: int,
              filter_ids: Optional[List] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_features: Query feature vector(s) - shape (n, dimension) or (dimension,)
            top_k: Number of results to return
            filter_ids: Optional list of product IDs to filter results
        
        Returns:
            Tuple of (distances, indices) where:
            - distances: Similarity scores (higher is better)
            - indices: FAISS index positions
        """
        if self.index is None:
            raise RuntimeError("FAISS index not initialized. Call _ensure_index_exists() first.")
        
        # Ensure query is 2D
        if len(query_features.shape) == 1:
            query_features = query_features.reshape(1, -1)
        
        # Ensure float32
        query_features = query_features.astype('float32')
        
        # Normalize query vectors
        faiss.normalize_L2(query_features)
        
        # Search
        distances, indices = self.index.search(query_features, top_k)
        
        # Apply filters if provided
        if filter_ids is not None:
            # Convert product IDs to FAISS indices
            id_to_idx = {pid: idx for idx, pid in enumerate(self.product_ids)}
            filter_indices = set(id_to_idx.get(pid, -1) for pid in filter_ids)
            filter_indices.discard(-1)  # Remove invalid IDs
            
            # Filter results
            filtered_distances = []
            filtered_indices = []
            for dist_row, idx_row in zip(distances, indices):
                filtered_dist = []
                filtered_idx = []
                for dist, idx in zip(dist_row, idx_row):
                    if idx in filter_indices:
                        filtered_dist.append(dist)
                        filtered_idx.append(idx)
                filtered_distances.append(filtered_dist)
                filtered_indices.append(filtered_idx)
            
            # Pad to top_k
            max_len = max(len(row) for row in filtered_distances) if filtered_distances else 0
            if max_len > 0:
                for i in range(len(filtered_distances)):
                    while len(filtered_distances[i]) < max_len:
                        filtered_distances[i].append(-1.0)
                        filtered_indices[i].append(-1)
                distances = np.array(filtered_distances)[:, :top_k]
                indices = np.array(filtered_indices)[:, :top_k]
            else:
                # No matches
                distances = np.full((len(query_features), top_k), -1.0)
                indices = np.full((len(query_features), top_k), -1, dtype=np.int64)
        
        return distances, indices
    
    def get_product_ids(self, indices: np.ndarray) -> List[str]:
        """
        Convert FAISS indices to product IDs.
        
        Args:
            indices: FAISS index positions
        
        Returns:
            List of product IDs
        """
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        
        product_ids = []
        for idx in indices:
            if 0 <= idx < len(self.product_ids):
                product_ids.append(self.product_ids[idx])
            else:
                product_ids.append(None)
        
        return product_ids
    
    def save_index(self):
        """Save FAISS index and ID mapping to disk"""
        if self.index is None:
            return
        
        # Save index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save ID mapping
        with open(self.ids_path, 'wb') as f:
            pickle.dump(self.product_ids, f)
    
    def load_index(self):
        """Load FAISS index and ID mapping from disk"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        
        # Load index
        self.index = faiss.read_index(str(self.index_path))
        
        # Load ID mapping
        if self.ids_path.exists():
            with open(self.ids_path, 'rb') as f:
                self.product_ids = pickle.load(f)
        else:
            # Fallback: generate IDs from index size
            self.product_ids = [str(i) for i in range(self.index.ntotal)]
    
    def is_initialized(self) -> bool:
        """Check if index is initialized"""
        return self.index is not None and len(self.product_ids) > 0
    
    def get_index_size(self) -> int:
        """Get number of vectors in index"""
        if self.index is None:
            return 0
        return self.index.ntotal

