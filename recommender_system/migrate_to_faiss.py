"""
Migration Script: Convert .npy features to FAISS index
This script manually migrates existing NumPy feature files to FAISS index.
Note: This is optional - the system will auto-migrate on first run if needed.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vector_db import VectorDB
from path_utils import find_file, get_faiss_index_dir


def migrate_to_faiss(npy_path: str = None, metadata_path: str = None):
    """
    Migrate .npy features to FAISS index.
    
    Args:
        npy_path: Path to resnet50_features_pca512.npy (auto-detects if None)
        metadata_path: Path to resnet50_metadata.csv (auto-detects if None)
    """
    print("=" * 60)
    print("FAISS Migration Script")
    print("=" * 60)
    
    # Find files if not provided
    if npy_path is None:
        npy_file = find_file("resnet50_features_pca512.npy")
        if npy_file:
            npy_path = str(npy_file)
        else:
            print("Error: Could not find resnet50_features_pca512.npy")
            print("Please provide the path manually or ensure the file exists.")
            return False
    
    if metadata_path is None:
        metadata_file = find_file("resnet50_metadata.csv")
        if metadata_file:
            metadata_path = str(metadata_file)
        else:
            print("Error: Could not find resnet50_metadata.csv")
            print("Please provide the path manually or ensure the file exists.")
            return False
    
    # Check if files exist
    if not os.path.exists(npy_path):
        print(f"Error: Feature file not found: {npy_path}")
        return False
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        return False
    
    print(f"\nFeature file: {npy_path}")
    print(f"Metadata file: {metadata_path}")
    
    # Check if index already exists
    index_dir = get_faiss_index_dir()
    index_path = index_dir / "product_index.faiss"
    
    if index_path.exists():
        response = input(f"\nFAISS index already exists at {index_path}\n"
                        "Do you want to overwrite it? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Migration cancelled.")
            return False
    
    # Create VectorDB and migrate
    try:
        print("\nCreating VectorDB...")
        vector_db = VectorDB(auto_init=False)  # Don't auto-init, we'll do it manually
        
        print("Migrating features to FAISS index...")
        vector_db._create_index_from_npy(npy_path, metadata_path)
        
        print("\n" + "=" * 60)
        print("✓ Migration completed successfully!")
        print(f"FAISS index saved to: {index_path}")
        print(f"Product IDs mapping saved to: {index_dir / 'product_ids.pkl'}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nError during migration: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate .npy features to FAISS index"
    )
    parser.add_argument(
        "--npy",
        type=str,
        help="Path to resnet50_features_pca512.npy (auto-detects if not provided)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to resnet50_metadata.csv (auto-detects if not provided)"
    )
    
    args = parser.parse_args()
    
    success = migrate_to_faiss(args.npy, args.metadata)
    sys.exit(0 if success else 1)

