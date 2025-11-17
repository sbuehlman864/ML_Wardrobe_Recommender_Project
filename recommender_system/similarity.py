"""
Similarity Matching Engine
Calculates similarity between user wardrobe and products using multiple strategies
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import os
from path_utils import find_file


class SimilarityMatcher:
    """Match user wardrobe features to product features"""
    
    def __init__(self, 
                 product_features_path: str = "../extracted_features/resnet50_features_pca512.npy",
                 product_metadata_path: str = "../extracted_features/resnet50_metadata.csv"):
        """
        Initialize similarity matcher.
        
        Args:
            product_features_path: Path to product feature matrix
            product_metadata_path: Path to product metadata CSV
        """
        # Load product features
        self.product_features = self._load_product_features(product_features_path)
        
        # Load product metadata
        self.product_metadata = self._load_product_metadata(product_metadata_path)
        
        print(f"Loaded {len(self.product_features)} products")
        print(f"Feature dimension: {self.product_features.shape[1]}")
    
    def _load_product_features(self, features_path: str) -> np.ndarray:
        """Load product feature matrix"""
        # Extract filename from path
        filename = os.path.basename(features_path) if os.path.sep in features_path else features_path
        if not filename.endswith('.npy'):
            filename = "resnet50_features_pca512.npy"
        
        # Try to find the file using path_utils
        features_file = find_file(filename)
        
        if features_file and features_file.exists():
            print(f"Loading product features from: {features_file}")
            features = np.load(str(features_file))
            return features
        
        # Fallback: try the provided path directly
        if os.path.exists(features_path):
            print(f"Loading product features from: {features_path}")
            features = np.load(features_path)
            return features
        
        raise FileNotFoundError(
            f"Product features not found. Looking for: {filename}\n"
            "Please ensure resnet50_features_pca512.npy exists in extracted_features/ directory"
        )
    
    def _load_product_metadata(self, metadata_path: str) -> pd.DataFrame:
        """Load product metadata"""
        # Extract filename from path
        filename = os.path.basename(metadata_path) if os.path.sep in metadata_path else metadata_path
        if not filename.endswith('.csv'):
            filename = "resnet50_metadata.csv"
        
        # Try to find the file using path_utils
        metadata_file = find_file(filename)
        
        if metadata_file and metadata_file.exists():
            print(f"Loading product metadata from: {metadata_file}")
            try:
                metadata = pd.read_csv(str(metadata_file), on_bad_lines='skip', encoding='utf-8')
            except TypeError:
                # Fallback for older pandas versions
                try:
                    metadata = pd.read_csv(str(metadata_file), error_bad_lines=False, warn_bad_lines=True, encoding='utf-8')
                except TypeError:
                    metadata = pd.read_csv(str(metadata_file), error_bad_lines=False, encoding='utf-8')
            return metadata
        
        # Fallback: try the provided path directly
        if os.path.exists(metadata_path):
            print(f"Loading product metadata from: {metadata_path}")
            try:
                metadata = pd.read_csv(metadata_path, on_bad_lines='skip', encoding='utf-8')
            except TypeError:
                try:
                    metadata = pd.read_csv(metadata_path, error_bad_lines=False, warn_bad_lines=True, encoding='utf-8')
                except TypeError:
                    metadata = pd.read_csv(metadata_path, error_bad_lines=False, encoding='utf-8')
            return metadata
        
        raise FileNotFoundError(
            f"Product metadata not found. Looking for: {filename}\n"
            "Please ensure resnet50_metadata.csv exists in extracted_features/ directory"
        )
    
    def calculate_cosine_similarity(self, 
                                   user_features: np.ndarray,
                                   product_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate cosine similarity between user features and product features.
        
        Args:
            user_features: User feature vector(s) - shape (n, 512) or (512,)
            product_features: Product feature matrix - shape (m, 512). If None, uses all products.
        
        Returns:
            Similarity scores - shape (n, m) or (m,)
        """
        if product_features is None:
            product_features = self.product_features
        
        # Ensure user_features is 2D
        if len(user_features.shape) == 1:
            user_features = user_features.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_features, product_features)
        
        # If single user feature, return 1D array
        if similarities.shape[0] == 1:
            return similarities[0]
        
        return similarities
    
    def find_similar_products(self,
                             user_features: np.ndarray,
                             top_k: int = 10,
                             min_similarity: float = 0.0,
                             filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Find top K similar products to user features.
        
        Args:
            user_features: User feature vector(s) - shape (512,) or (n, 512)
            top_k: Number of top similar products to return
            min_similarity: Minimum similarity threshold
            filters: Optional filters dict with keys like:
                - gender: List of genders to include
                - masterCategory: List of categories
                - articleType: List of article types
                - baseColour: List of colors
                - season: List of seasons
                - usage: List of usage types
        
        Returns:
            DataFrame with top K similar products and their metadata
        """
        # Calculate similarities
        similarities = self.calculate_cosine_similarity(user_features)
        
        # Handle multiple user features (average similarity)
        if len(similarities.shape) > 1:
            similarities = similarities.mean(axis=0)
        
        # Apply filters
        if filters:
            mask = self._create_filter_mask(filters)
            # Set filtered products to -1 (will be below min_similarity)
            similarities[~mask] = -1
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        # Filter by minimum similarity
        valid_mask = top_similarities >= min_similarity
        top_indices = top_indices[valid_mask]
        top_similarities = top_similarities[valid_mask]
        
        # Get product metadata
        results = self.product_metadata.iloc[top_indices].copy()
        results['similarity_score'] = top_similarities
        
        # Sort by similarity
        results = results.sort_values('similarity_score', ascending=False)
        
        return results
    
    def _create_filter_mask(self, filters: Dict) -> np.ndarray:
        """Create boolean mask for filtering products"""
        mask = np.ones(len(self.product_metadata), dtype=bool)
        
        for key, value in filters.items():
            if key in self.product_metadata.columns:
                if isinstance(value, list):
                    mask &= self.product_metadata[key].isin(value)
                else:
                    mask &= (self.product_metadata[key] == value)
        
        return mask
    
    def find_complementary_products(self,
                                   user_wardrobe_features: np.ndarray,
                                   user_wardrobe_metadata: pd.DataFrame,
                                   top_k: int = 10,
                                   filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Find complementary products (different categories that go well together).
        
        Args:
            user_wardrobe_features: Features of user's wardrobe items
            user_wardrobe_metadata: Metadata of user's wardrobe items
            top_k: Number of recommendations
            filters: Optional filters dict (e.g., gender, category, color)
        
        Returns:
            DataFrame with complementary product recommendations
        """
        # Analyze user's wardrobe
        user_categories = user_wardrobe_metadata.get('articleType', pd.Series()).value_counts()
        user_colors = user_wardrobe_metadata.get('baseColour', pd.Series()).value_counts()
        user_usage = user_wardrobe_metadata.get('usage', pd.Series()).value_counts()
        
        # Define complementary categories
        complementary_map = {
            'Tshirts': ['Jeans', 'Shorts', 'Casual Shoes', 'Belts'],
            'Shirts': ['Jeans', 'Formal Shoes', 'Belts'],
            'Jeans': ['Tshirts', 'Shirts', 'Casual Shoes', 'Belts'],
            'Tops': ['Jeans', 'Shorts', 'Flats', 'Handbags'],
            'Casual Shoes': ['Jeans', 'Tshirts', 'Shorts'],
            'Formal Shoes': ['Jeans', 'Shirts'],
        }
        
        # Find complementary categories
        target_categories = []
        for category in user_categories.index:
            if category in complementary_map:
                target_categories.extend(complementary_map[category])
        
        # Remove duplicates and categories user already has
        target_categories = list(set(target_categories) - set(user_categories.index))
        
        # Filter products by complementary categories
        internal_filters = {'articleType': target_categories} if target_categories else {}
        
        # Also match style (usage type)
        if len(user_usage) > 0:
            dominant_usage = user_usage.index[0]
            if 'usage' not in internal_filters:
                internal_filters['usage'] = [dominant_usage]
        
        # Merge with external filters (e.g., gender)
        if filters:
            for key, value in filters.items():
                if key in internal_filters:
                    # If both have same key, intersect the values
                    if isinstance(internal_filters[key], list) and isinstance(value, list):
                        internal_filters[key] = list(set(internal_filters[key]) & set(value))
                    elif isinstance(value, list):
                        internal_filters[key] = value
                else:
                    internal_filters[key] = value
        
        # Find similar products in complementary categories
        # Average user wardrobe features
        avg_user_features = user_wardrobe_features.mean(axis=0) if len(user_wardrobe_features.shape) > 1 else user_wardrobe_features
        
        results = self.find_similar_products(
            avg_user_features,
            top_k=top_k * 2,  # Get more, then filter
            filters=internal_filters if internal_filters else None
        )
        
        # Ensure diversity (not all same category)
        if len(results) > top_k:
            # Group by category and take top from each
            diverse_results = []
            for category in results['articleType'].unique()[:5]:  # Max 5 categories
                category_items = results[results['articleType'] == category].head(2)
                diverse_results.append(category_items)
            
            if diverse_results:
                results = pd.concat(diverse_results).head(top_k)
        
        return results.head(top_k)
    
    def find_by_category_expansion(self,
                                  user_wardrobe_metadata: pd.DataFrame,
                                  top_k: int = 10,
                                  filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Find products to fill wardrobe gaps (category expansion).
        
        Args:
            user_wardrobe_metadata: Metadata of user's wardrobe
            top_k: Number of recommendations
            filters: Optional filters dict (e.g., gender, category, color)
        
        Returns:
            DataFrame with products to fill gaps
        """
        # Analyze wardrobe distribution
        category_counts = user_wardrobe_metadata.get('articleType', pd.Series()).value_counts()
        
        # Find underrepresented categories
        all_categories = self.product_metadata['articleType'].value_counts()
        user_categories = set(category_counts.index)
        
        # Find categories user doesn't have or has few of
        gap_categories = []
        for category in all_categories.index:
            if category not in user_categories:
                gap_categories.append(category)
            elif category_counts.get(category, 0) < 2:  # Has less than 2 items
                gap_categories.append(category)
        
        # Get top categories to recommend
        target_categories = gap_categories[:5]  # Top 5 gap categories
        
        if not target_categories:
            # If no gaps, recommend diverse categories
            target_categories = list(all_categories.index[:10])
        
        # Filter and get diverse products
        internal_filters = {'articleType': target_categories}
        
        # Merge with external filters (e.g., gender)
        if filters:
            for key, value in filters.items():
                if key in internal_filters:
                    # If both have same key, intersect the values
                    if isinstance(internal_filters[key], list) and isinstance(value, list):
                        internal_filters[key] = list(set(internal_filters[key]) & set(value))
                    elif isinstance(value, list):
                        internal_filters[key] = value
                else:
                    internal_filters[key] = value
        
        # Get products from gap categories with filters applied
        mask = self._create_filter_mask(internal_filters)
        filtered_products = self.product_metadata[mask]
        
        # Sample diverse products
        results = filtered_products.groupby('articleType').head(2).head(top_k)
        
        # Add dummy similarity scores (not based on visual similarity)
        results['similarity_score'] = 0.5  # Neutral score for category expansion
        
        return results


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score (0 to 1)
    """
    # Ensure vectors are 1D
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)
    
    # Calculate magnitudes
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Cosine similarity
    similarity = dot_product / (norm1 * norm2)
    
    return float(similarity)


if __name__ == "__main__":
    # Test similarity matching
    import sys
    
    if len(sys.argv) > 1:
        # Test with sample features
        test_features = np.random.rand(512)  # Random test features
        
        print("Testing similarity matcher...")
        matcher = SimilarityMatcher()
        
        # Find similar products
        results = matcher.find_similar_products(test_features, top_k=5)
        print("\nTop 5 similar products:")
        print(results[['id', 'articleType', 'baseColour', 'similarity_score']].head())
        
        print("\n✓ Similarity matching test successful!")
    else:
        print("Usage: python similarity.py")
        print("Or use in code:")
        print("  from similarity import SimilarityMatcher")
        print("  matcher = SimilarityMatcher()")
        print("  results = matcher.find_similar_products(user_features, top_k=10)")

