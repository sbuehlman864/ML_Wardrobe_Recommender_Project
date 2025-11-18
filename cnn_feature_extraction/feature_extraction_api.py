from tkinter import N
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from typing import Union, List, Optional

from image_loader import ImageLoader
from cnn_model import FeatureExtractor

class FeatureExtractionAPI:
    """
    API for extracting features from images
    For use by other team members (KNN, Clustering)
    """

    def __init__(
        self,
        features_dir: str = "extracted_features",
        model_name: str = "resnet50"
    ):
        """
        Initialize API
        
        Args:
            features_dir: Folder with extracted features
            model_name: Model name
        """
        self.features_dir = Path(features_dir)
        self.model_name = model_name
        
        print("="*70)
        print("Feature Extraction API - CNN Module")
        print("="*70)

        print("\nLoading extracted features...")
        self.precomputed_features = self._load_precomputed_features()
        self.metadata = self._load_metadata()

        print(f"✓ Loaded {len(self.precomputed_features['normalized'])} feature vectors")
        print(f"✓ Available versions: normalized (2048), pca512 (512)")

        self.image_loader = None
        self.feature_extractor = None
        self.pca_model = None

    def _load_precomputed_features(self) -> dict:
        """Loads all versions of precomputed features"""
        features = {}

        path_orig = self.features_dir / f"{self.model_name}_features.npy"
        if path_orig.exists():
            features['original'] = np.load(path_orig)

        path_norm = self.features_dir / f"{self.model_name}_features_normalized.npy"
        if path_norm.exists():
            features['normalized'] = np.load(path_norm)
        
        # PCA
        path_pca = self.features_dir / f'{self.model_name}_features_pca512.npy'
        if path_pca.exists():
            features['pca512'] = np.load(path_pca)
        
        return features

    def _load_metadata(self) -> pd.DataFrame:
        """Loads metadata"""
        path = self.features_dir / f'{self.model_name}_metadata.csv'
        return pd.read_csv(path)

    def get_all_features(self, version: str = 'normalized') -> np.ndarray:
        """
        Returns all extracted feature vectors
        
        Args:
            version: 'original', 'normalized', or 'pca512'
        
        Returns:
            numpy array (N, feature_dim)
        """
        if version not in self.precomputed_features:
            raise ValueError(f"Version '{version}' not available. Available: {list(self.precomputed_features.keys())}")
        
        return self.precomputed_features[version]

    def get_metadata(self) -> pd.DataFrame:
        """
        Returns metadata of all images
        
        Returns:
            DataFrame with id, gender, masterCategory, articleType, baseColour, etc.
        """
        return self.metadata.copy()

    def get_features_by_ids(self, ids: List[int], version: str = 'normalized') -> np.ndarray:
        """
        Returns feature vectors for specific IDs
        
        Args:
            ids: List of image IDs
            version: Features version
        
        Returns:
            numpy array (len(ids), feature_dim)
        """
        features = self.get_all_features(version)

        indices = []
        for img_id in ids:
            idx = self.metadata[self.metadata['id'] == img_id].index
            if len(idx) > 0:
                indices.append(idx[0])

        if not indices:
            raise ValueError("None of the provided IDs found in dataset")
        
        return features[indices]

    def get_features_by_category(
        self,
        category: str,
        category_type: str = 'articleType',
        version: str = 'normalized'
    ) -> tuple:
        """
        Returns features for a specific category
        
        Args:
            category: Category name (e.g., 'Shirts', 'Jeans')
            category_type: Category type ('articleType', 'masterCategory', 'gender')
            version: Features version
        
        Returns:
            tuple: (features, ids, metadata)
        """
        mask = self.metadata[category_type] == category
        filtered_metadata = self.metadata[mask]

        if len(filtered_metadata) == 0:
            raise ValueError(f"Category '{category}' not found in {category_type}")


        indices = filtered_metadata.index.tolist()
        features = self.get_all_features(version)[indices]
        ids = filtered_metadata['id'].tolist()
        
        return features, ids, filtered_metadata

    def extract_features_from_new_image(
        self,
        image_id: int,
        apply_normalization: bool = True,
        apply_pca: bool = False
    ) -> np.ndarray:
        """
        Extracts features from a new image (not in precomputed)
        
        Args:
            image_id: Image ID
            apply_normalization: Apply L2 normalization
            apply_pca: Apply PCA (512 components)
        
        Returns:
            numpy array feature vector
        """
        if self.image_loader is None:
            self.image_loader = ImageLoader(target_size=(224, 224))
            self.feature_extractor = FeatureExtractor(model_name = self.model_name)
        
        img = self.image_loader.preprocess_image(image_id, model_type=self.model_name)

        if img is None:
            raise ValueError(f"Image {image_id} not found")
        
        features = self.feature_extractor.extract_single_feature(img)
        
        if apply_normalization:
            from sklearn.preprocessing import normalize
            features = normalize(features.reshape(1, -1), norm='l2')[0]
        
        # PCA
        if apply_pca:
            if self.pca_model is None:
                pca_path = self.features_dir / f'{self.model_name}_pca512_model.pkl'
                with open(pca_path, 'rb') as f:
                    self.pca_model = pickle.load(f)
            features = self.pca_model.transform(features.reshape(1, -1))[0]
        
        return features

    def save_features_for_team(self, output_dir: str = 'features_for_team'):
        """
        Saves features in a convenient format for the team
        
        Args:
            output_dir: Folder for saving
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print("Saving data for team...")
        print(f"{'='*70}")
        
        # 1. Features (normalized) - for KNN
        np.save(output_path / 'features_normalized.npy', self.precomputed_features['normalized'])
        print(f"✓ features_normalized.npy - (41802, 2048) for full accuracy")
        
        # 2. Features (PCA) - for speed
        np.save(output_path / 'features_pca512.npy', self.precomputed_features['pca512'])
        print(f"✓ features_pca512.npy - (41802, 512) for speed")
        
        # 3. Metadata
        self.metadata.to_csv(output_path / 'metadata.csv', index=False)
        print(f"✓ metadata.csv - image metadata")
        
        # 4. Mapping ID -> index
        id_to_index = {int(row['id']): idx for idx, row in self.metadata.iterrows()}
        with open(output_path / 'id_to_index.json', 'w') as f:
            json.dump(id_to_index, f)
        print(f"✓ id_to_index.json - mapping of image IDs to indices")
        
        # 5. Features information
        info = {
            'total_images': len(self.metadata),
            'feature_dimensions': {
                'normalized': self.precomputed_features['normalized'].shape[1],
                'pca512': self.precomputed_features['pca512'].shape[1]
            },
            'categories': {
                'masterCategory': self.metadata['masterCategory'].unique().tolist(),
                'articleType': self.metadata['articleType'].unique().tolist()[:20], 
                'gender': self.metadata['gender'].unique().tolist()
            },
            'model_used': self.model_name,
            'normalization': 'L2 (unit vectors)',
            'pca_explained_variance': 0.9425  
        }
        
        with open(output_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        print(f"✓ dataset_info.json - dataset information")
        
        print(f"\n{'='*70}")
        print(f"All files saved in: {output_path}")
        print(f"{'='*70}")


def demo_api():
    """
    Demonstration of API usage
    """

    api = FeatureExtractionAPI()
    
    print("\n" + "="*70)
    print("API DEMONSTRATION")
    print("="*70)
    
    # 1. Get all features
    print("\n1. Get all features (normalized):")
    all_features = api.get_all_features(version='normalized')
    print(f"   Shape: {all_features.shape}")
    print(f"   Type: {all_features.dtype}")
    
    # 2. Get metadata
    print("\n2. Get metadata:")
    metadata = api.get_metadata()
    print(f"   Rows: {len(metadata)}")
    print(f"   Columns: {list(metadata.columns)}")
    
    # 3. Features for specific IDs
    print("\n3. Features for specific IDs:")
    test_ids = [15970, 39386, 59263]
    features = api.get_features_by_ids(test_ids, version='normalized')
    print(f"   Requested IDs: {test_ids}")
    print(f"   Features shape: {features.shape}")
    
    # 4. Features by category
    print("\n4. Features by category (Shirts):")
    shirt_features, shirt_ids, shirt_meta = api.get_features_by_category('Shirts')
    print(f"   Found: {len(shirt_ids)} images")
    print(f"   Features shape: {shirt_features.shape}")
    print(f"   First 5 IDs: {shirt_ids[:5]}")
    
    # 5. Save for team
    print("\n5. Saving data for team:")
    api.save_features_for_team(output_dir='features_for_team')
    
    print("\n" + "="*70)
    print("API is ready for use!")
    print("="*70)


if __name__ == "__main__":
    demo_api()