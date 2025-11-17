"""
Test Script for Recommendation System
Simple script to test and get recommendations
"""

import sys
import os
from pathlib import Path
from recommender import Recommender
import pandas as pd

def main():
    """Test the recommendation system"""
    
    # Default wardrobe folder
    default_wardrobe = "../Wardrobe_upload_system/wardrobe_storage/jashwanth"
    
    # Get wardrobe folder from command line or use default
    if len(sys.argv) > 1:
        wardrobe_folder = sys.argv[1]
    else:
        wardrobe_folder = default_wardrobe
    
    # Check if folder exists
    if not os.path.exists(wardrobe_folder):
        print(f"Error: Wardrobe folder not found: {wardrobe_folder}")
        print(f"\nUsage: python test_recommender.py [wardrobe_folder]")
        print(f"Example: python test_recommender.py {default_wardrobe}")
        return
    
    # Get all images from folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(wardrobe_folder).glob(f"*{ext}"))
    
    image_paths = [str(p) for p in image_paths]
    
    if not image_paths:
        print(f"No images found in {wardrobe_folder}")
        print("Please add some images to your wardrobe folder first.")
        return
    
    print("=" * 70)
    print("Wardrobe Recommendation System")
    print("=" * 70)
    print(f"Wardrobe folder: {wardrobe_folder}")
    print(f"Found {len(image_paths)} images in wardrobe")
    print("=" * 70)
    
    # Initialize recommender
    print("\nInitializing recommender system...")
    try:
        recommender = Recommender()
    except Exception as e:
        print(f"Error initializing recommender: {e}")
        return
    
    # Get recommendations
    print("\n" + "=" * 70)
    print("Getting Recommendations (Hybrid Strategy)")
    print("=" * 70)
    print("This may take a minute...")
    
    try:
        recommendations = recommender.get_recommendations(
            user_wardrobe_paths=image_paths,
            strategy='hybrid',  # Best approach!
            top_k=20,
            filters=None  # Can add filters like {'gender': ['Men'], 'season': ['Summer']}
        )
        
        if len(recommendations) == 0:
            print("No recommendations found. Try adjusting filters or adding more wardrobe items.")
            return
        
        # Display results
        print("\n" + "=" * 70)
        print(f"Top {len(recommendations)} Recommendations")
        print("=" * 70)
        
        for idx, (_, item) in enumerate(recommendations.iterrows(), 1):
            print(f"\n{idx}. {item.get('productDisplayName', 'N/A')}")
            print(f"   ID: {item.get('id', 'N/A')}")
            print(f"   Type: {item.get('articleType', 'N/A')}")
            print(f"   Color: {item.get('baseColour', 'N/A')}")
            print(f"   Category: {item.get('masterCategory', 'N/A')}")
            print(f"   Similarity: {item.get('similarity_score', 0):.3f}")
            print(f"   Reason: {item.get('recommendation_reason', 'N/A')}")
        
        # Save to CSV
        output_file = "recommendations.csv"
        recommendations.to_csv(output_file, index=False)
        print(f"\n" + "=" * 70)
        print(f"Recommendations saved to: {output_file}")
        print("=" * 70)
        
        # Summary statistics
        print("\nRecommendation Summary:")
        print(f"- Total recommendations: {len(recommendations)}")
        print(f"- Unique categories: {recommendations['articleType'].nunique()}")
        print(f"- Average similarity: {recommendations['similarity_score'].mean():.3f}")
        print(f"- Category distribution:")
        print(recommendations['articleType'].value_counts().head(10).to_string())
        
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

