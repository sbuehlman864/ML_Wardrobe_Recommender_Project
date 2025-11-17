"""
Recommendation Engine
Hybrid recommendation system combining multiple strategies
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from similarity import SimilarityMatcher
from feature_extractor import FeatureExtractor
import os

class Recommender:
    """Hybrid recommendation engine"""
    
    def __init__(self,
                 feature_extractor: Optional[FeatureExtractor] = None,
                 similarity_matcher: Optional[SimilarityMatcher] = None):
        """
        Initialize recommender.
        
        Args:
            feature_extractor: FeatureExtractor instance (creates if None)
            similarity_matcher: SimilarityMatcher instance (creates if None)
        """
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.similarity_matcher = similarity_matcher or SimilarityMatcher()
        
        print("✓ Recommender initialized")
    
    def get_recommendations(self,
                           user_wardrobe_paths: List[str],
                           strategy: str = 'hybrid',
                           top_k: int = 20,
                           filters: Optional[Dict] = None,
                           weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Get recommendations based on user's wardrobe.
        
        Args:
            user_wardrobe_paths: List of paths to user's wardrobe images
            strategy: Recommendation strategy:
                - 'similar': Visual similarity only
                - 'complementary': Complementary items only
                - 'category_expansion': Fill wardrobe gaps only
                - 'hybrid': Combine all strategies (recommended)
            top_k: Number of recommendations to return
            filters: Optional filters (gender, category, color, etc.)
            weights: Optional weights for hybrid strategy:
                - similar: Weight for visual similarity (default: 0.4)
                - complementary: Weight for complementary items (default: 0.3)
                - category_expansion: Weight for category expansion (default: 0.2)
                - diversity: Weight for diversity boost (default: 0.1)
        
        Returns:
            DataFrame with recommendations and scores
        """
        if not user_wardrobe_paths:
            raise ValueError("User wardrobe paths cannot be empty")
        
        # Extract features from user wardrobe
        print(f"Extracting features from {len(user_wardrobe_paths)} wardrobe items...")
        user_features = self.feature_extractor.batch_extract_features(user_wardrobe_paths)
        
        # Get metadata for user wardrobe (if available)
        # For now, we'll infer from filenames or use defaults
        user_metadata = self._infer_wardrobe_metadata(user_wardrobe_paths)
        
        # Apply strategy
        if strategy == 'similar':
            results = self._get_similar_recommendations(
                user_features, top_k, filters
            )
        elif strategy == 'complementary':
            results = self._get_complementary_recommendations(
                user_features, user_metadata, top_k, filters
            )
        elif strategy == 'category_expansion':
            results = self._get_category_expansion_recommendations(
                user_metadata, top_k, filters
            )
        elif strategy == 'hybrid':
            results = self._get_hybrid_recommendations(
                user_features, user_metadata, top_k, filters, weights
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Add recommendation reasons
        results = self._add_recommendation_reasons(results, strategy)
        
        return results
    
    def _get_similar_recommendations(self,
                                    user_features: np.ndarray,
                                    top_k: int,
                                    filters: Optional[Dict]) -> pd.DataFrame:
        """Get visually similar recommendations"""
        return self.similarity_matcher.find_similar_products(
            user_features, top_k=top_k, filters=filters
        )
    
    def _get_complementary_recommendations(self,
                                          user_features: np.ndarray,
                                          user_metadata: pd.DataFrame,
                                          top_k: int,
                                          filters: Optional[Dict]) -> pd.DataFrame:
        """Get complementary item recommendations"""
        results = self.similarity_matcher.find_complementary_products(
            user_features, user_metadata, top_k=top_k, filters=filters
        )
        
        return results.head(top_k)
    
    def _get_category_expansion_recommendations(self,
                                               user_metadata: pd.DataFrame,
                                               top_k: int,
                                               filters: Optional[Dict]) -> pd.DataFrame:
        """Get category expansion recommendations"""
        results = self.similarity_matcher.find_by_category_expansion(
            user_metadata, top_k=top_k, filters=filters
        )
        
        return results.head(top_k)
    
    def _get_hybrid_recommendations(self,
                                   user_features: np.ndarray,
                                   user_metadata: pd.DataFrame,
                                   top_k: int,
                                   filters: Optional[Dict],
                                   weights: Optional[Dict[str, float]]) -> pd.DataFrame:
        """
        Get hybrid recommendations combining multiple strategies.
        """
        # Default weights
        default_weights = {
            'similar': 0.4,
            'complementary': 0.3,
            'category_expansion': 0.2,
            'diversity': 0.1
        }
        if weights:
            default_weights.update(weights)
        
        all_recommendations = []
        
        # Strategy 1: Visual Similarity (40%)
        similar_count = int(top_k * default_weights['similar'])
        if similar_count > 0:
            similar = self._get_similar_recommendations(
                user_features, similar_count, filters
            )
            similar['strategy'] = 'similar'
            similar['strategy_weight'] = default_weights['similar']
            all_recommendations.append(similar)
        
        # Strategy 2: Complementary Items (30%)
        comp_count = int(top_k * default_weights['complementary'])
        if comp_count > 0:
            complementary = self._get_complementary_recommendations(
                user_features, user_metadata, comp_count, filters
            )
            complementary['strategy'] = 'complementary'
            complementary['strategy_weight'] = default_weights['complementary']
            all_recommendations.append(complementary)
        
        # Strategy 3: Category Expansion (20%)
        cat_count = int(top_k * default_weights['category_expansion'])
        if cat_count > 0:
            category_exp = self._get_category_expansion_recommendations(
                user_metadata, cat_count, filters
            )
            category_exp['strategy'] = 'category_expansion'
            category_exp['strategy_weight'] = default_weights['category_expansion']
            all_recommendations.append(category_exp)
        
        # Combine all recommendations
        if not all_recommendations:
            return pd.DataFrame()
        
        combined = pd.concat(all_recommendations, ignore_index=True)
        
        # Remove duplicates (keep first occurrence)
        combined = combined.drop_duplicates(subset=['id'], keep='first')
        
        # Re-rank by weighted score
        combined['weighted_score'] = (
            combined['similarity_score'] * combined['strategy_weight']
        )
        combined = combined.sort_values('weighted_score', ascending=False)
        
        # Ensure diversity (not all same category)
        diverse_results = self._ensure_diversity(combined, top_k)
        
        return diverse_results.head(top_k)
    
    def _ensure_diversity(self, results: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """Ensure recommendations are diverse across categories"""
        if len(results) <= top_k:
            return results
        
        diverse = []
        categories_seen = set()
        max_per_category = max(1, top_k // 5)  # Max items per category
        
        for _, row in results.iterrows():
            category = row.get('articleType', 'Unknown')
            
            # Add if category not seen too many times
            if category not in categories_seen or \
               len([r for r in diverse if r.get('articleType') == category]) < max_per_category:
                diverse.append(row)
                categories_seen.add(category)
            
            if len(diverse) >= top_k:
                break
        
        # Fill remaining slots with best remaining items
        if len(diverse) < top_k:
            remaining = results[~results['id'].isin([r['id'] for r in diverse])]
            diverse.extend(remaining.head(top_k - len(diverse)).to_dict('records'))
        
        return pd.DataFrame(diverse)
    
    def _infer_wardrobe_metadata(self, wardrobe_paths: List[str]) -> pd.DataFrame:
        """Infer metadata from wardrobe image paths/filenames"""
        # Simple inference - can be enhanced
        metadata_list = []
        
        for path in wardrobe_paths:
            filename = os.path.basename(path)
            
            # Try to extract article type from filename
            article_type = 'Clothing'  # Default
            for art_type in ['Tshirts', 'Shirts', 'Jeans', 'Shoes', 'Watches']:
                if art_type.lower() in filename.lower():
                    article_type = art_type
                    break
            
            metadata_list.append({
                'image_path': path,
                'articleType': article_type,
                'baseColour': 'Unknown',  # Could be enhanced with color detection
                'usage': 'Casual',  # Default
            })
        
        return pd.DataFrame(metadata_list)
    
    def _apply_filters(self, results: pd.DataFrame, filters: Dict) -> pd.Series:
        """Apply filters to results"""
        mask = pd.Series([True] * len(results), index=results.index)
        
        for key, value in filters.items():
            if key in results.columns:
                if isinstance(value, list):
                    mask &= results[key].isin(value)
                else:
                    mask &= (results[key] == value)
        
        return mask
    
    def _add_recommendation_reasons(self, results: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Add explanation for why each item was recommended"""
        reasons = []
        
        for _, row in results.iterrows():
            strategy_type = row.get('strategy', strategy)
            
            if strategy_type == 'similar':
                score = row.get('similarity_score', 0)
                reason = f"Visually similar to your wardrobe (similarity: {score:.2f})"
            elif strategy_type == 'complementary':
                reason = f"Complements your {row.get('articleType', 'items')}"
            elif strategy_type == 'category_expansion':
                reason = f"Fills wardrobe gap in {row.get('articleType', 'category')}"
            else:
                reason = "Recommended based on your wardrobe"
            
            reasons.append(reason)
        
        results['recommendation_reason'] = reasons
        return results
    
    def get_complementary_items(self,
                               user_wardrobe_paths: List[str],
                               top_k: int = 10) -> pd.DataFrame:
        """Get complementary items for user's wardrobe"""
        user_features = self.feature_extractor.batch_extract_features(user_wardrobe_paths)
        user_metadata = self._infer_wardrobe_metadata(user_wardrobe_paths)
        
        return self.similarity_matcher.find_complementary_products(
            user_features, user_metadata, top_k=top_k
        )
    
    def get_outfit_suggestions(self,
                              user_wardrobe_paths: List[str],
                              num_outfits: int = 5) -> List[Dict]:
        """
        Suggest complete outfits from user's wardrobe + recommendations.
        
        Returns:
            List of outfit dictionaries with items and recommendations
        """
        # Get complementary recommendations
        recommendations = self.get_complementary_items(user_wardrobe_paths, top_k=20)
        
        # Group by category
        outfits = []
        categories = ['Tops', 'Bottoms', 'Shoes', 'Accessories']
        
        for i in range(num_outfits):
            outfit = {
                'outfit_id': i + 1,
                'items': [],
                'recommendations': []
            }
            
            # Select items from different categories
            for category in categories:
                category_items = recommendations[
                    recommendations['articleType'].str.contains(category, case=False, na=False)
                ]
                if len(category_items) > 0:
                    item = category_items.iloc[i % len(category_items)]
                    outfit['recommendations'].append({
                        'id': item['id'],
                        'articleType': item['articleType'],
                        'productDisplayName': item.get('productDisplayName', 'N/A')
                    })
            
            if outfit['recommendations']:
                outfits.append(outfit)
        
        return outfits


if __name__ == "__main__":
    # Test recommender
    import sys
    
    if len(sys.argv) > 1:
        wardrobe_folder = sys.argv[1]
        
        # Get all images from folder
        from pathlib import Path
        image_paths = list(Path(wardrobe_folder).glob("*.jpg")) + \
                     list(Path(wardrobe_folder).glob("*.png"))
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"No images found in {wardrobe_folder}")
            sys.exit(1)
        
        print(f"Testing recommender with {len(image_paths)} wardrobe items...")
        
        recommender = Recommender()
        
        # Get hybrid recommendations
        recommendations = recommender.get_recommendations(
            image_paths,
            strategy='hybrid',
            top_k=10
        )
        
        print("\nTop 10 Recommendations:")
        print(recommendations[['id', 'articleType', 'baseColour', 'similarity_score', 'recommendation_reason']].head(10))
        
        print("\n✓ Recommendation test successful!")
    else:
        print("Usage: python recommender.py <wardrobe_folder>")
        print("Example: python recommender.py ../Wardrobe_upload_system/wardrobe_storage/jashwanth")

