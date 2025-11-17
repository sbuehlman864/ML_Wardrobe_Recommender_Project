"""
Image Preprocessing Pipeline
Preprocesses images for ResNet50 feature extraction
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Union, Tuple, Optional, List
from pathlib import Path

# ImageNet normalization constants (used by ResNet50)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ResNet50 input size
RESNET_INPUT_SIZE = 224


def preprocess_image(image_path: Union[str, Path], 
                    target_size: Tuple[int, int] = (224, 224),
                    normalize: bool = True) -> torch.Tensor:
    """
    Preprocess image for ResNet50 feature extraction.
    
    Args:
        image_path: Path to image file (str or Path)
        target_size: Target size (width, height). Default: (224, 224)
        normalize: Whether to apply ImageNet normalization. Default: True
    
    Returns:
        Preprocessed image tensor ready for ResNet50
    """
    # Convert to string if Path object
    image_path = str(image_path) if isinstance(image_path, Path) else image_path
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")
    
    # Create transform pipeline
    transform_list = [
        transforms.Resize(256),  # Resize to 256 (slightly larger)
        transforms.CenterCrop(target_size[0]),  # Center crop to target size
        transforms.ToTensor(),  # Convert to tensor [0, 1]
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )
    
    transform = transforms.Compose(transform_list)
    
    # Apply transforms
    tensor = transform(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor


def preprocess_image_opencv(image_path: Union[str, Path],
                           target_size: Tuple[int, int] = (224, 224),
                           normalize: bool = True) -> np.ndarray:
    """
    Preprocess image using OpenCV (alternative method).
    
    Args:
        image_path: Path to image file (str or Path)
        target_size: Target size (width, height). Default: (224, 224)
        normalize: Whether to normalize. Default: True
    
    Returns:
        Preprocessed image array
    """
    # Convert to string if Path object
    image_path = str(image_path) if isinstance(image_path, Path) else image_path
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Convert to float and normalize
    image = image.astype(np.float32) / 255.0
    
    if normalize:
        # Apply ImageNet normalization
        image = (image - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    
    # Convert to CHW format (Channels, Height, Width)
    image = np.transpose(image, (2, 0, 1))
    
    return image


def check_image_quality(image_path: Union[str, Path]) -> dict:
    """
    Check image quality and validity.
    
    Args:
        image_path: Path to image file (str or Path)
    
    Returns:
        Dictionary with quality metrics
    """
    # Convert to string if Path object
    image_path = str(image_path) if isinstance(image_path, Path) else image_path
    
    try:
        image = Image.open(image_path)
        img_array = np.array(image)
        
        quality_info = {
            'valid': True,
            'width': image.width,
            'height': image.height,
            'channels': len(image.getbands()),
            'format': image.format,
            'mode': image.mode,
            'size_bytes': image.size[0] * image.size[1] * len(image.getbands()),
            'is_rgb': image.mode == 'RGB',
            'aspect_ratio': image.width / image.height if image.height > 0 else 0,
        }
        
        # Check if image is too small
        quality_info['too_small'] = image.width < 50 or image.height < 50
        
        # Check if image is corrupted
        try:
            image.verify()
            quality_info['corrupted'] = False
        except Exception:
            quality_info['corrupted'] = True
            quality_info['valid'] = False
        
        return quality_info
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


def handle_image_orientation(image_path: Union[str, Path]) -> Image.Image:
    """
    Handle image orientation (EXIF data).
    
    Args:
        image_path: Path to image file (str or Path)
    
    Returns:
        Correctly oriented PIL Image
    """
    from PIL.ExifTags import ORIENTATION
    
    # Convert to string if Path object
    image_path = str(image_path) if isinstance(image_path, Path) else image_path
    
    image = Image.open(image_path)
    
    try:
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(ORIENTATION)
            
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, TypeError):
        # No EXIF data or orientation tag
        pass
    
    return image.convert('RGB')


def batch_preprocess_images(image_paths: List[Union[str, Path]],
                           target_size: Tuple[int, int] = (224, 224),
                           normalize: bool = True) -> torch.Tensor:
    """
    Preprocess multiple images in batch.
    
    Args:
        image_paths: List of image file paths (str or Path)
        target_size: Target size (width, height)
        normalize: Whether to apply ImageNet normalization
    
    Returns:
        Batch tensor of preprocessed images
    """
    tensors = []
    
    for image_path in image_paths:
        try:
            tensor = preprocess_image(image_path, target_size, normalize)
            tensors.append(tensor.squeeze(0))  # Remove batch dim
        except Exception as e:
            print(f"Warning: Skipping {image_path}: {e}")
            continue
    
    if not tensors:
        raise ValueError("No valid images to preprocess")
    
    # Stack into batch
    batch_tensor = torch.stack(tensors)
    
    return batch_tensor


def remove_background(image_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> np.ndarray:
    """
    Remove background from image (optional enhancement).
    Uses simple color-based background removal.
    
    Args:
        image_path: Path to input image (str or Path)
        output_path: Optional path to save result (str or Path)
    
    Returns:
        Image with background removed (RGBA)
    """
    # Convert to string if Path object
    image_path = str(image_path) if isinstance(image_path, Path) else image_path
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image {image_path}")
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Simple background removal (white/light background)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create mask (assuming light background)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations to clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Create RGBA image
    image_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
    image_rgba[:, :, 3] = mask
    
    # Save if output path provided
    if output_path:
        output_path = str(output_path) if isinstance(output_path, Path) else output_path
        cv2.imwrite(output_path, cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA))
    
    return image_rgba


if __name__ == "__main__":
    # Test preprocessing
    import sys
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        
        print(f"Testing preprocessing on: {test_image}")
        
        # Check quality
        quality = check_image_quality(test_image)
        print(f"Quality check: {quality}")
        
        # Preprocess
        tensor = preprocess_image(test_image)
        print(f"Preprocessed tensor shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")
        print(f"Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        
        print("✓ Preprocessing test successful!")

