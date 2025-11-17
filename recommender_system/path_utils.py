"""
Path Utilities for Cross-Platform Compatibility
Provides robust path resolution that works on Windows, macOS, and Linux
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union


def get_project_root() -> Path:
    """
    Get the project root directory (parent of recommender_system).
    
    Returns:
        Path to project root
    """
    # Get the directory of this file
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    
    # If we're in recommender_system, go up one level
    if current_dir.name == "recommender_system":
        return current_dir.parent
    
    # Otherwise, try to find project root by looking for common markers
    markers = [
        "ML_Wardrobe_Recommender_Project-main 2",
        "feature_extraction",
        "Wardrobe_upload_system",
        "EDA"
    ]
    
    # Walk up the directory tree
    for parent in current_dir.parents:
        if any((parent / marker).exists() for marker in markers):
            return parent
    
    # Fallback: assume we're in recommender_system
    return current_dir.parent


def get_recommender_system_dir() -> Path:
    """
    Get the recommender_system directory.
    
    Returns:
        Path to recommender_system directory
    """
    return Path(__file__).resolve().parent


def find_file(filename: str, 
              search_dirs: Optional[List[Union[str, Path]]] = None,
              relative_to: Optional[Path] = None) -> Optional[Path]:
    """
    Find a file in multiple possible locations.
    
    Args:
        filename: Name of the file to find
        search_dirs: List of directories to search (if None, uses common locations)
        relative_to: Base path for relative searches (default: project root)
    
    Returns:
        Path to file if found, None otherwise
    """
    if relative_to is None:
        relative_to = get_project_root()
    
    # Default search locations
    if search_dirs is None:
        search_dirs = [
            relative_to / "extracted_features",
            relative_to / "feature_extraction",
            relative_to / "recommender_system",
            get_recommender_system_dir(),
            Path.cwd(),
            Path.cwd() / "extracted_features",
            Path.cwd() / "feature_extraction",
        ]
    
    # Convert all to Path objects and resolve
    search_paths = []
    for search_dir in search_dirs:
        if isinstance(search_dir, str):
            search_dir = Path(search_dir)
        
        # Try absolute path
        if search_dir.is_absolute():
            search_paths.append(search_dir.resolve())
        else:
            # Try relative to relative_to
            search_paths.append((relative_to / search_dir).resolve())
            # Try relative to current working directory
            search_paths.append((Path.cwd() / search_dir).resolve())
            # Try relative to script location
            search_paths.append((get_recommender_system_dir() / search_dir).resolve())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in search_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    
    # Search for file
    for search_dir in unique_paths:
        file_path = search_dir / filename
        if file_path.exists() and file_path.is_file():
            return file_path.resolve()
    
    return None


def find_directory(dirname: str,
                   search_dirs: Optional[List[Union[str, Path]]] = None,
                   relative_to: Optional[Path] = None) -> Optional[Path]:
    """
    Find a directory in multiple possible locations.
    
    Args:
        dirname: Name of the directory to find
        search_dirs: List of parent directories to search
        relative_to: Base path for relative searches
    
    Returns:
        Path to directory if found, None otherwise
    """
    if relative_to is None:
        relative_to = get_project_root()
    
    if search_dirs is None:
        search_dirs = [
            relative_to,
            get_recommender_system_dir(),
            Path.cwd(),
        ]
    
    # Convert all to Path objects
    search_paths = []
    for search_dir in search_dirs:
        if isinstance(search_dir, str):
            search_dir = Path(search_dir)
        
        if search_dir.is_absolute():
            search_paths.append(search_dir.resolve())
        else:
            search_paths.append((relative_to / search_dir).resolve())
            search_paths.append((Path.cwd() / search_dir).resolve())
            search_paths.append((get_recommender_system_dir() / search_dir).resolve())
    
    # Remove duplicates
    seen = set()
    unique_paths = []
    for path in search_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    
    # Search for directory
    for search_dir in unique_paths:
        dir_path = search_dir / dirname
        if dir_path.exists() and dir_path.is_dir():
            return dir_path.resolve()
    
    return None


def get_wardrobe_storage_dir() -> Path:
    """
    Get the wardrobe storage directory.
    Creates it if it doesn't exist.
    
    Returns:
        Path to wardrobe storage directory
    """
    # Try multiple possible locations
    possible_locations = [
        get_recommender_system_dir() / "wardrobe_storage",
        get_project_root() / "Wardrobe_upload_system" / "wardrobe_storage",
        Path.cwd() / "wardrobe_storage",
    ]
    
    for location in possible_locations:
        if location.exists():
            return location.resolve()
    
    # Create in recommender_system directory (most likely location)
    storage_dir = get_recommender_system_dir() / "wardrobe_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir.resolve()


def get_product_images_dir() -> Optional[Path]:
    """
    Find the directory containing product images.
    
    Returns:
        Path to images directory if found, None otherwise
    """
    project_root = get_project_root()
    
    possible_dirs = [
        project_root / "fashion-product-images-small" / "images",
        project_root / "YOLO_training" / "data" / "raw" / "fashion-product-images-small" / "images",
        project_root / "feature_extraction" / "images",
        project_root / "EDA" / "images",
        # Kaggle dataset locations (cross-platform)
        Path.home() / ".cache" / "kagglehub" / "datasets" / "paramaggarwal" / "fashion-product-images-dataset" / "fashion-dataset" / "images",
        Path.home() / ".cache" / "kagglehub" / "datasets" / "paramaggarwal" / "fashion-product-images-small" / "images",
        # Windows alternative
        Path(os.environ.get("LOCALAPPDATA", "")) / "kagglehub" / "datasets" / "paramaggarwal" / "fashion-product-images-dataset" / "fashion-dataset" / "images",
    ]
    
    for img_dir in possible_dirs:
        if img_dir.exists() and img_dir.is_dir():
            return img_dir.resolve()
    
    return None


def normalize_path(path: Union[str, Path]) -> Path:
    """
    Normalize a path for cross-platform compatibility.
    
    Args:
        path: Path string or Path object
    
    Returns:
        Normalized Path object
    """
    if isinstance(path, str):
        path = Path(path)
    
    # Expand user home directory
    if path.parts and path.parts[0] == "~":
        path = path.expanduser()
    
    # Resolve to absolute path
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        # If resolution fails (e.g., path doesn't exist), return as-is but expanded
        return path.expanduser().absolute()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to directory
    
    Returns:
        Path object to the directory
    """
    path = normalize_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device() -> str:
    """
    Detect the best available device for PyTorch.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'


if __name__ == "__main__":
    # Test path utilities
    print("Project root:", get_project_root())
    print("Recommender system dir:", get_recommender_system_dir())
    print("Wardrobe storage dir:", get_wardrobe_storage_dir())
    print("Product images dir:", get_product_images_dir())
    print("Device:", get_device())
    
    # Test file finding
    test_file = find_file("recommender.py")
    print(f"Found recommender.py: {test_file}")

