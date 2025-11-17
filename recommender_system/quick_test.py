"""
Quick Test - Get recommendations for a single image
"""

import sys
from recommender import Recommender

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py <image_path>")
        print("Example: python quick_test.py ../Wardrobe_upload_system/wardrobe_storage/jashwanth/image.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Testing with image: {image_path}")
    print("Initializing...")
    
    recommender = Recommender()
    
    print("Getting recommendations...")
    recommendations = recommender.get_recommendations(
        user_wardrobe_paths=[image_path],
        strategy='hybrid',
        top_k=10
    )
    
    print(f"\nTop 10 Recommendations:")
    print(recommendations[['id', 'articleType', 'baseColour', 'similarity_score']].to_string())

