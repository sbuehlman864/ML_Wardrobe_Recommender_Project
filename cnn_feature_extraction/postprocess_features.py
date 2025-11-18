from pyexpat import features
import numpy as np
from numpy.random import normal
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import pickle
import json
from typing import Tuple, Optional


class FeaturePostprocessor:
    """
    Class for postprocessing extracted features:
    - L2 normalization
    - PCA for dimensionality reduction (optional)
    """
    def __init__(self, features_dir: str = 'extracted_features', model_name: str = 'resnet50'):

        """
        Initialization
        
        Args:
            features_dir: Folder with extracted features
            model_name: Model name
        """


        self.features_dir = Path(features_dir)
        self.model_name = model_name

        print("="*70)
        print("STEP 7: Postprocessing feature vectors")
        print("="*70)

        print("\n1. Loading extracted features...")
        self.features = self._load_features()
        self.metadata = self._load_metadata()

        print(f"   ✓ Features shape: {self.features.shape}")
        print(f"   ✓ Metadata records: {len(self.metadata)}")

        print("\n2. Original features statistics:")
        print(f"   Mean: {self.features.mean():.4f}")
        print(f"   Std: {self.features.std():.4f}")
        print(f"   Min: {self.features.min():.4f}")
        print(f"   Max: {self.features.max():.4f}")

    def _load_features(self) -> np.ndarray:
        """Loads feature vectors"""
        features_path = self.features_dir / f'{self.model_name}_features.npy'
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        return np.load(features_path)

    def _load_metadata(self) -> pd.DataFrame:
        """Loads metadata"""
        metadata_path = self.features_dir / f'{self.model_name}_metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        return pd.read_csv(metadata_path)

    def normalize_features(self, norm: str = 'l2') -> np.ndarray:
        """
        Normalizes feature vectors
        
        Args:
            norm: Normalization type ('l2', 'l1', 'max')
        
        Returns:
            Normalized features
        """
        print(f"\n3. Normalizing features ({norm})...")

        normalized_features = normalize(self.features, norm=norm, axis=1)

        print(f"   ✓ Normalization completed")
        print(f"   Shape: {normalized_features.shape}")

        norms = norms = np.linalg.norm(normalized_features, axis=1)
        print(f"   Vector norms - Mean: {norms.mean():.4f}, Std: {norms.std():.6f}")

        output_path = self.features_dir / f'{self.model_name}_features_normalized.npy'
        np.save(output_path, normalized_features)
        print(f"   ✓ Saved to: {output_path}")
        
        return normalized_features

    def apply_pca(
        self,
        n_components: int = 512,
        use_normalized: bool = True
    ) -> Tuple[np.ndarray, PCA]:
        """
        Applies PCA for dimensionality reduction
        
        Args:
            n_components: Number of components to keep
            use_normalized: Use normalized features
        
        Returns:
            Tuple: (reduced_features, pca_model)
        """
        print(f"\n4. Applying PCA (n_components={n_components})...")

        if use_normalized:
            normalized_path = self.features_dir / f'{self.model_name}_features_normalized.npy'
            if normalized_path.exists():
                input_features = np.load(normalized_path)
                print(f"   Using normalized features")
            else:
                input_features = self.features
                print(f"   Normalized not found, using original")
        else:
            input_features = self.features

        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(input_features)

        print(f"   ✓ PCA completed")
        print(f"   Original dimension: {input_features.shape[1]}")
        print(f"   New dimension: {reduced_features.shape[1]}")
        print(f"   Explained variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")

        output_path = self.features_dir / f'{self.model_name}_features_pca{n_components}.npy'
        np.save(output_path, reduced_features)
        print(f"   ✓ Reduced features saved: {output_path}")

        pca_model_path = self.features_dir / f'{self.model_name}_pca{n_components}_model.pkl'
        with open(pca_model_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"   ✓ PCA model saved: {pca_model_path}")
        
        pca_info = {
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_explained_variance': float(pca.explained_variance_ratio_.sum()),
            'singular_values': pca.singular_values_.tolist()[:10]
        }
        
        pca_info_path = self.features_dir / f'{self.model_name}_pca{n_components}_info.json'
        with open(pca_info_path, 'w') as f:
            json.dump(pca_info, f, indent=2)
        print(f"   ✓ PCA info saved: {pca_info_path}")
        
        return reduced_features, pca

    def create_summary(self):
        """
        Creates a summary of all processed files
        """
        print("\n" + "="*70)
        print("5. Summary of processed files")
        print("="*70)
        
        files_info = []
        

        original_path = self.features_dir / f'{self.model_name}_features.npy'
        if original_path.exists():
            size_mb = original_path.stat().st_size / (1024 * 1024)
            files_info.append({
                'file': original_path.name,
                'shape': self.features.shape,
                'size_mb': size_mb,
                'description': 'Original feature vectors'
            })
        

        normalized_path = self.features_dir / f'{self.model_name}_features_normalized.npy'
        if normalized_path.exists():
            features_norm = np.load(normalized_path)
            size_mb = normalized_path.stat().st_size / (1024 * 1024)
            files_info.append({
                'file': normalized_path.name,
                'shape': features_norm.shape,
                'size_mb': size_mb,
                'description': 'L2 normalized (for cosine similarity)'
            })
        

        for pca_file in self.features_dir.glob(f'{self.model_name}_features_pca*.npy'):
            features_pca = np.load(pca_file)
            size_mb = pca_file.stat().st_size / (1024 * 1024)
            n_comp = features_pca.shape[1]
            files_info.append({
                'file': pca_file.name,
                'shape': features_pca.shape,
                'size_mb': size_mb,
                'description': f'PCA reduced ({n_comp} components)'
            })
        

        metadata_path = self.features_dir / f'{self.model_name}_metadata.csv'
        if metadata_path.exists():
            size_mb = metadata_path.stat().st_size / (1024 * 1024)
            files_info.append({
                'file': metadata_path.name,
                'shape': (len(self.metadata), len(self.metadata.columns)),
                'size_mb': size_mb,
                'description': 'Image metadata'
            })
        

        print("\n{:<50} {:<20} {:<10} {}".format("File", "Shape", "Size", "Description"))
        print("-" * 110)
        for info in files_info:
            print("{:<50} {:<20} {:<10.2f} MB {}".format(
                info['file'], 
                str(info['shape']), 
                info['size_mb'],
                info['description']
            ))
        
        print("\n" + "="*70)
        print("Postprocessing completed!")
        print("="*70)

def main():
    """
    Main function for postprocessing
    """

    processor = FeaturePostprocessor(
        features_dir='extracted_features',
        model_name='resnet50'
    )
    

    normalized_features = processor.normalize_features(norm='l2')
    

    print("\n" + "="*70)
    print("Apply PCA?")
    print("="*70)
    print("PCA will reduce dimensionality from 2048 to a smaller number.")
    print("Pros: faster KNN, less memory")
    print("Cons: small accuracy loss")
    print("\nRecommended values: 256, 512, 1024")
    
    pca_components = 512
    print(f"\nUsing n_components={pca_components}")
    
    reduced_features, pca_model = processor.apply_pca(
        n_components=pca_components,
        use_normalized=True
    )
    
    processor.create_summary()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR NEXT STEPS")
    print("="*70)
    print("\nFor KNN and similarity search use:")
    print("  ✓ resnet50_features_normalized.npy (for full accuracy)")
    print("  or")
    print(f"  ✓ resnet50_features_pca{pca_components}.npy (for speed)")
    print("\nTogether with:")
    print("  ✓ resnet50_metadata.csv (for linking IDs with categories)")


if __name__ == "__main__":
    main()