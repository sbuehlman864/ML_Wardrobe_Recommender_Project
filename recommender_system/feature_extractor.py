"""
Feature Extraction Module
Extracts 512-dimensional features from user wardrobe images using ResNet50 + PCA
"""

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Union, List, Optional
from preprocessing import preprocess_image, batch_preprocess_images
from path_utils import find_file, get_device


class FeatureExtractor:
    """Extract features from images using ResNet50 + PCA"""
    
    def __init__(self, 
                 pca_model_path: str = "../extracted_features/resnet50_pca512_model.pkl",
                 device: Optional[str] = None):
        """
        Initialize feature extractor.
        
        Args:
            pca_model_path: Path to trained PCA model
            device: Device to use ('cuda', 'mps', 'cpu'). Auto-detects if None
        """
        # Detect device
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load ResNet50 model
        self.model = self._load_resnet50()
        self.model.eval()
        self.model.to(self.device)
        
        # Load PCA model
        self.pca_model = self._load_pca_model(pca_model_path)
        
        print("✓ Feature extractor initialized")
    
    def _load_resnet50(self) -> nn.Module:
        """Load pre-trained ResNet50 model"""
        print("Loading ResNet50 model...")
        
        # First, check if model exists in cache
        cache_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth")
        if os.path.exists(cache_path):
            print(f"Found cached model at {cache_path}")
            try:
                model = models.resnet50(weights=None)
                model.load_state_dict(torch.load(cache_path, map_location='cpu'))
                print("✓ Loaded ResNet50 from cache")
            except Exception as e:
                print(f"Warning: Error loading from cache: {e}")
                print("Attempting to download...")
                model = self._download_resnet50()
        else:
            # Try to download
            print("Model not in cache, attempting to download...")
            model = self._download_resnet50()
        
        # Remove final classification layer
        # Keep everything except the last fully connected layer
        model = nn.Sequential(*list(model.children())[:-1])
        
        return model
    
    def _download_resnet50(self) -> nn.Module:
        """Download ResNet50 model with fallback options"""
        try:
            # Try using the newer weights API
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            print("✓ Downloaded ResNet50 using weights API")
            return model
        except (ImportError, AttributeError):
            # Fallback to deprecated pretrained parameter
            try:
                model = models.resnet50(pretrained=True)
                print("✓ Downloaded ResNet50 using pretrained parameter")
                return model
            except Exception as e:
                # If download fails, provide helpful error message
                cache_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth")
                raise RuntimeError(
                    f"Could not download ResNet50 model: {e}\n\n"
                    "Solutions:\n"
                    "1. Run: ./download_resnet50.sh (to download manually)\n"
                    "2. Or fix SSL certificates: /Applications/Python\\ 3.x/Install\\ Certificates.command\n"
                    "3. Or manually download from:\n"
                    "   https://download.pytorch.org/models/resnet50-0676ba61.pth\n"
                    f"   and save to: {cache_path}"
                )
    
    def _load_pca_model(self, pca_path: str) -> object:
        """Load trained PCA model"""
        print(f"Loading PCA model...")
        
        # Extract filename from path
        filename = os.path.basename(pca_path) if os.path.sep in pca_path else pca_path
        if not filename.endswith('.pkl'):
            filename = "resnet50_pca512_model.pkl"
        
        # Try to find the file using path_utils
        pca_file = find_file(filename)
        
        if pca_file and pca_file.exists():
            print(f"Loading PCA model from: {pca_file}")
            with open(pca_file, 'rb') as f:
                pca_model = pickle.load(f)
            print("✓ PCA model loaded")
            return pca_model
        
        # Fallback: try the provided path directly
        if os.path.exists(pca_path):
            print(f"Loading PCA model from: {pca_path}")
            with open(pca_path, 'rb') as f:
                pca_model = pickle.load(f)
            print("✓ PCA model loaded")
            return pca_model
        
        raise FileNotFoundError(
            f"PCA model not found. Looking for: {filename}\n"
            "Please ensure resnet50_pca512_model.pkl exists in extracted_features/ directory"
        )
    
    def extract_features_2048(self, image_path: Union[str, torch.Tensor]) -> np.ndarray:
        """
        Extract 2048-dimensional features from image (before PCA).
        
        Args:
            image_path: Path to image file or preprocessed tensor
        
        Returns:
            2048-dimensional feature vector
        """
        # Preprocess if path provided
        if isinstance(image_path, str):
            image_tensor = preprocess_image(image_path)
        else:
            image_tensor = image_path
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)
            # Global average pooling (already done by ResNet50)
            features = features.squeeze()  # Remove batch and spatial dims
            
            # Handle batch case
            if len(features.shape) > 1:
                features = features.view(features.size(0), -1)
            else:
                features = features.view(1, -1)
            
            # Convert to numpy
            features_np = features.cpu().numpy()
            
            # Clip extreme values to prevent numerical issues in PCA
            # This helps avoid overflow/underflow warnings
            features_np = np.clip(features_np, -100, 100)
        
        return features_np
    
    def extract_features(self, image_path: Union[str, torch.Tensor]) -> np.ndarray:
        """
        Extract 512-dimensional features from image (after PCA).
        
        Args:
            image_path: Path to image file or preprocessed tensor
        
        Returns:
            512-dimensional feature vector
        """
        # Extract 2048-dim features
        features_2048 = self.extract_features_2048(image_path)
        
        # Apply PCA transformation
        features_512 = self.pca_model.transform(features_2048)
        
        return features_512
    
    def batch_extract_features(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract features from multiple images in batch.
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            Array of 512-dimensional feature vectors (N x 512)
        """
        # Preprocess all images
        batch_tensor = batch_preprocess_images(image_paths)
        batch_tensor = batch_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(batch_tensor)
            # Reshape: (batch, 2048, 1, 1) -> (batch, 2048)
            features = features.squeeze()
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            features = features.view(features.size(0), -1)
            
            # Convert to numpy
            features_2048 = features.cpu().numpy()
            
            # Clip extreme values to prevent numerical issues in PCA
            features_2048 = np.clip(features_2048, -100, 100)
        
        # Apply PCA
        features_512 = self.pca_model.transform(features_2048)
        
        return features_512
    
    def extract_from_wardrobe_folder(self, wardrobe_folder: str) -> dict:
        """
        Extract features from all images in a wardrobe folder.
        
        Args:
            wardrobe_folder: Path to folder containing wardrobe images
        
        Returns:
            Dictionary mapping image paths to feature vectors
        """
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(wardrobe_folder).glob(f'*{ext}'))
        
        if not image_paths:
            print(f"No images found in {wardrobe_folder}")
            return {}
        
        print(f"Found {len(image_paths)} images in wardrobe folder")
        
        # Extract features
        features_dict = {}
        
        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_paths_str = [str(p) for p in batch_paths]
            
            try:
                features = self.batch_extract_features(batch_paths_str)
                
                for path, feature in zip(batch_paths_str, features):
                    features_dict[path] = feature
                
                print(f"Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images...")
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
        
        print(f"✓ Extracted features from {len(features_dict)} images")
        return features_dict


def extract_user_features(image_path: str,
                          pca_model_path: str = "../extracted_features/resnet50_pca512_model.pkl") -> np.ndarray:
    """
    Convenience function to extract features from a single image.
    
    Args:
        image_path: Path to image file
        pca_model_path: Path to PCA model
    
    Returns:
        512-dimensional feature vector
    """
    extractor = FeatureExtractor(pca_model_path)
    return extractor.extract_features(image_path)


if __name__ == "__main__":
    # Test feature extraction
    import sys
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        
        print(f"Testing feature extraction on: {test_image}")
        
        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract_features(test_image)
        
        print(f"Feature vector shape: {features.shape}")
        print(f"Feature vector dtype: {features.dtype}")
        print(f"Feature vector range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"Feature vector mean: {features.mean():.3f}")
        print(f"Feature vector std: {features.std():.3f}")
        
        print("✓ Feature extraction test successful!")
    else:
        print("Usage: python feature_extractor.py <image_path>")
        print("Or test with wardrobe folder:")
        print("  extractor = FeatureExtractor()")
        print("  features = extractor.extract_from_wardrobe_folder('path/to/wardrobe')")

