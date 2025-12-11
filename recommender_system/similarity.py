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
from vector_db import VectorDB


class SimilarityMatcher:
    """Match user wardrobe features to product features"""
    
    def __init__(self, 
                 product_features_path: str = "../extracted_features/resnet50_features_pca512.npy",
                 product_metadata_path: str = "../extracted_features/resnet50_metadata.csv",
                 vector_db: Optional[VectorDB] = None):
        """
        Initialize similarity matcher.
        
        Args:
            product_features_path: Path to product feature matrix (for fallback)
            product_metadata_path: Path to product metadata CSV
            vector_db: Optional VectorDB instance (creates if None)
        """
        # Try to initialize VectorDB (will auto-create if needed)
        self.use_faiss = False
        self.product_features = None
        
        try:
            if vector_db is None:
                self.vector_db = VectorDB(auto_init=True)
            else:
                self.vector_db = vector_db
            
            if self.vector_db.is_initialized():
                self.use_faiss = True
                print("Using FAISS vector database")
            else:
                raise RuntimeError("VectorDB not initialized")
        except (FileNotFoundError, RuntimeError, ImportError) as e:
            # Fallback to NumPy if FAISS fails
            print(f"FAISS not available, falling back to NumPy: {e}")
            self.use_faiss = False
            self.vector_db = None
            # Load product features for fallback
            self.product_features = self._load_product_features(product_features_path)
        
        # Load product metadata (always needed)
        self.product_metadata = self._load_product_metadata(product_metadata_path)
        
        if self.use_faiss:
            print(f"FAISS index contains {self.vector_db.get_index_size()} products")
        else:
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
            "Please ensure resnet50_features_pca512.npy exists in feature_extraction/ directory"
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
            "Please ensure resnet50_metadata.csv exists in feature_extraction/ directory"
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
        if self.use_faiss:
            # Use FAISS search (returns top results, but we need all for compatibility)
            # For now, search for all products (top_k = index size)
            top_k = self.vector_db.get_index_size()
            distances, indices = self.vector_db.search(user_features, top_k=top_k)
            
            # Convert to full similarity array
            # FAISS returns distances (inner product), which are already similarity scores
            if len(distances.shape) == 1:
                # Single query
                similarities = np.zeros(self.vector_db.get_index_size())
                for dist, idx in zip(distances, indices):
                    if idx >= 0:
                        similarities[idx] = dist
                return similarities
            else:
                # Multiple queries - return as-is for now
                return distances
        else:
            # Fallback to NumPy
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
                             filters: Optional[Dict] = None,
                             diversity: bool = True) -> pd.DataFrame:
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
            diversity: If True, ensures diversity across categories (default: True)
        
        Returns:
            DataFrame with top K similar products and their metadata
        """
        if self.use_faiss:
            # Use FAISS for efficient search
            # Handle multiple user features (average them)
            if len(user_features.shape) > 1:
                avg_features = user_features.mean(axis=0)
            else:
                avg_features = user_features
            
            # Search with FAISS
            # Search for more results to enable diversity selection
            search_k = min(top_k * 10, self.vector_db.get_index_size()) if diversity else (min(top_k * 5, self.vector_db.get_index_size()) if filters else top_k)
            distances, indices = self.vector_db.search(
                avg_features.reshape(1, -1), 
                top_k=search_k
            )
            
            # Get product IDs from FAISS indices
            faiss_indices = indices[0]  # First (and only) query
            product_ids = self.vector_db.get_product_ids(faiss_indices)
            similarities = distances[0]  # First query results
            
            # Create mapping from product ID to similarity and FAISS index
            id_to_sim = {}
            id_to_faiss_idx = {}
            for pid, sim, fidx in zip(product_ids, similarities, faiss_indices):
                if pid is not None and fidx >= 0:
                    id_to_sim[str(pid)] = sim
                    id_to_faiss_idx[str(pid)] = fidx
            
            # Apply filters to metadata
            if filters:
                mask = self._create_filter_mask(filters)
                filtered_metadata = self.product_metadata[mask].copy()
            else:
                filtered_metadata = self.product_metadata.copy()
            
            # Filter by minimum similarity and map to metadata
            results_list = []
            for _, row in filtered_metadata.iterrows():
                product_id = str(row['id'])
                if product_id in id_to_sim:
                    sim_score = id_to_sim[product_id]
                    if sim_score >= min_similarity:
                        row_copy = row.copy()
                        row_copy['similarity_score'] = sim_score
                        results_list.append(row_copy)
            
            if len(results_list) == 0:
                return pd.DataFrame()
            
            # Convert to DataFrame and sort
            results = pd.DataFrame(results_list)
            results = results.sort_values('similarity_score', ascending=False)
            
            # Apply diversity if requested
            if diversity and len(results) > top_k:
                results = self._apply_diversity_selection(results, top_k)
            
            return results.head(top_k)
        else:
            # Fallback to NumPy method
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
            
            # Get more candidates if diversity is enabled
            candidate_k = top_k * 10 if diversity else top_k * 5 if filters else top_k
            
            # Get top candidate indices
            top_indices = np.argsort(similarities)[::-1][:candidate_k]
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
            
            # Apply diversity if requested
            if diversity and len(results) > top_k:
                results = self._apply_diversity_selection(results, top_k)
            
            return results.head(top_k)
    
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
    
    def _apply_diversity_selection(self, results: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """
        Apply diversity selection to ensure variety in recommendations.
        Selects items from different categories and adds randomization for similar scores.
        """
        if len(results) <= top_k:
            return results
        
        # Add small random perturbation to scores within similar ranges
        # This helps break ties and introduce variety
        results = results.copy()
        max_score = results['similarity_score'].max()
        
        # Add random noise to scores (small, to not drastically change ranking)
        np.random.seed()  # Use current time as seed for variety
        noise = np.random.normal(0, max_score * 0.01, len(results))  # 1% noise
        results['diversity_score'] = results['similarity_score'] + noise
        
        # Group by category to ensure diversity
        if 'articleType' in results.columns:
            diverse_results = []
            categories_seen = set()
            max_per_category = max(1, top_k // 4)  # Max items per category (4 categories)
            
            # Sort by diversity score (similarity + noise)
            results_sorted = results.sort_values('diversity_score', ascending=False)
            
            for _, row in results_sorted.iterrows():
                category = str(row.get('articleType', 'Unknown'))
                category_count = sum(1 for r in diverse_results if str(r.get('articleType', 'Unknown')) == category)
                
                # Add if category not seen too many times, or if we need more items
                if category_count < max_per_category or len(diverse_results) < top_k // 2:
                    # Convert Series to dict to ensure consistent data type
                    diverse_results.append(row.to_dict())
                    categories_seen.add(category)
                
                if len(diverse_results) >= top_k:
                    break
            
            # Fill remaining slots with best remaining items (ensuring no duplicates)
            if len(diverse_results) < top_k:
                selected_ids = {r.get('id') for r in diverse_results}
                remaining = results_sorted[~results_sorted['id'].isin(selected_ids)]
                diverse_results.extend(remaining.head(top_k - len(diverse_results)).to_dict('records'))
            
            # Convert back to DataFrame and sort by original similarity score
            # All items should now be dictionaries, so DataFrame creation should work
            diverse_df = pd.DataFrame(diverse_results)
            diverse_df = diverse_df.sort_values('similarity_score', ascending=False)
            # Drop the temporary diversity_score column
            if 'diversity_score' in diverse_df.columns:
                diverse_df = diverse_df.drop('diversity_score', axis=1)
            return diverse_df.head(top_k)
        else:
            # If no category column, just add randomization to top items
            # Shuffle items with similar scores
            top_items = results.head(top_k * 2).copy()  # Get more candidates
            # Add small random noise for variety
            np.random.seed()  # Use current time as seed for variety
            max_score = top_items['similarity_score'].max()
            noise = np.random.normal(0, max_score * 0.01, len(top_items))
            top_items['diversity_score'] = top_items['similarity_score'] + noise
            top_items = top_items.sort_values('diversity_score', ascending=False)
            result = top_items.head(top_k).sort_values('similarity_score', ascending=False)
            if 'diversity_score' in result.columns:
                result = result.drop('diversity_score', axis=1)
            return result
    
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
        complementary_filters = {'articleType': target_categories} if target_categories else {}
        
        # Also match style (usage type)
        if len(user_usage) > 0:
            dominant_usage = user_usage.index[0]
            if 'usage' not in complementary_filters:
                complementary_filters['usage'] = [dominant_usage]
        
        # Merge with provided filters (e.g., gender)
        if filters:
            # Merge filters - provided filters take precedence for overlapping keys
            merged_filters = complementary_filters.copy()
            for key, value in filters.items():
                if key in merged_filters:
                    # If both have the same key, intersect the values
                    if isinstance(merged_filters[key], list) and isinstance(value, list):
                        merged_filters[key] = [v for v in merged_filters[key] if v in value]
                    else:
                        merged_filters[key] = value
                else:
                    merged_filters[key] = value
            filters = merged_filters
        else:
            filters = complementary_filters
        
        # Find similar products in complementary categories
        # Average user wardrobe features
        avg_user_features = user_wardrobe_features.mean(axis=0) if len(user_wardrobe_features.shape) > 1 else user_wardrobe_features
        
        results = self.find_similar_products(
            avg_user_features,
            top_k=top_k * 2,  # Get more, then filter
            filters=filters,
            diversity=True  # Enable diversity for complementary items
        )
        
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
        category_filters = {'articleType': target_categories}
        
        # Merge with provided filters (e.g., gender)
        if filters:
            # Merge filters - provided filters take precedence for overlapping keys
            merged_filters = category_filters.copy()
            for key, value in filters.items():
                if key in merged_filters:
                    # If both have the same key, intersect the values
                    if isinstance(merged_filters[key], list) and isinstance(value, list):
                        merged_filters[key] = [v for v in merged_filters[key] if v in value]
                    else:
                        merged_filters[key] = value
                else:
                    merged_filters[key] = value
            filters = merged_filters
        else:
            filters = category_filters
        
        # Apply filters to get products from gap categories
        filter_mask = self._create_filter_mask(filters)
        filtered_products = self.product_metadata[filter_mask].copy()
        
        # Add randomization for variety
        np.random.seed()  # Use current time as seed for variety
        filtered_products = filtered_products.sample(frac=1.0, random_state=None).reset_index(drop=True)
        
        # Sample diverse products from each category
        diverse_results = []
        category_counts = {}
        max_per_category = max(1, top_k // len(target_categories) if target_categories else top_k)
        
        for _, row in filtered_products.iterrows():
            category = str(row.get('articleType', 'Unknown'))
            category_count = category_counts.get(category, 0)
            
            if category_count < max_per_category or len(diverse_results) < top_k // 2:
                diverse_results.append(row)
                category_counts[category] = category_count + 1
            
            if len(diverse_results) >= top_k:
                break
        
        # Fill remaining slots if needed
        if len(diverse_results) < top_k:
            selected_ids = {r.get('id') for r in diverse_results}
            remaining = filtered_products[~filtered_products['id'].isin(selected_ids)]
            diverse_results.extend(remaining.head(top_k - len(diverse_results)).to_dict('records'))
        
        results = pd.DataFrame(diverse_results)
        
        # Add dummy similarity scores (not based on visual similarity)
        results['similarity_score'] = 0.5  # Neutral score for category expansion
        
        return results.head(top_k)


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

