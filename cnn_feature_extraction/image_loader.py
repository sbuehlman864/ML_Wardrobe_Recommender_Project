import os
import json
from keras.src.utils import mode_keys
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing import image as keras_image


class ImageLoader:
    """
    Class for loading and preprocessing images for CNN
    """

    def __init__(self, config_path: str = 'dataset_config.json', target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image loader
        
        Args:
            config_path: Path to configuration file with dataset paths
            target_size: Target size for images (width, height) for CNN
        """
        self.target_size = target_size

        with open(config_path, 'r') as f:
            config = json.load(f)

        self.images_folder = Path(config['images_folder'])
        self.dataset_path = Path(config['dataset_path'])

        print(f"ImageLoader initialized:")
        print(f"  - Images folder: {self.images_folder}")
        print(f"  - Target size: {target_size}")

    def load_image(self, image_id: int, return_pil: bool=False) -> Optional[np.ndarray]:
        """
        Loads a single image by ID
        
        Args:
            image_id: Image ID from dataset
            return_pil: If True, returns PIL Image instead of numpy array
            
        Returns:
            numpy array of image or None if not found
        """
        img_path = self.images_folder / f"{image_id}.jpg"

        if not img_path.exists():
            return None

        try:
            img = Image.open(img_path)

            if img.mode != "RGB":
                img = img.convert("RGB")

            if return_pil:
                return img

            img_array = np.array(img)
            return img_array

        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            return None


    def preprocess_image(self, image_id: int, model_type: str = 'resnet50') -> Optional[np.ndarray]:
        """
        Loads and preprocesses image for a specific CNN model
        
        Args:
            image_id: Image ID
            model_type: Model type ('resnet50', 'vgg16', 'efficientnet')
            
        Returns:
            Preprocessed image ready for CNN (shape: 224, 224, 3)
        """
        img_path = self.images_folder / f"{image_id}.jpg"

        if not img_path.exists():
            return None

        try:
            img = Image.open(img_path)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize(self.target_size, Image.LANCZOS)

            img_array = np.array(img, dtype=np.float32)

            if model_type == 'resnet50':
                img_array = resnet_preprocess(img_array)
            elif model_type == 'vgg16':
                from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
                img_array = vgg_preprocess(img_array)
            else:
                img_array = img_array / 255.0

            return img_array

        except Exception as e:
            print(f"Error preprocessing image {image_id}: {e}")
            return None

    def load_batch(self, image_ids: List[int], model_type: str = 'resnet50') -> Tuple[np.ndarray, List[int]]:
        """
        Loads a batch of images
        
        Args:
            image_ids: List of image IDs
            model_type: Model type for preprocessing
            
        Returns:
            Tuple: (batch_array, valid_ids)
                - batch_array: numpy array of shape (batch_size, 224, 224, 3)
                - valid_ids: list of IDs that were successfully loaded
        """
        batch_images = []
        valid_ids = []

        for img_id in image_ids:
            img = self.preprocess_image(img_id, model_type=model_type)
            if img is not None:
                batch_images.append(img)
                valid_ids.append(img_id)

        if not batch_images:
            return np.array([]), []

        batch_array = np.stack(batch_images, axis=0)

        return batch_array, valid_ids

    def visualize_preprocessing(self, image_id: int, save_path: Optional[str] = None):
        """
        Visualizes original and preprocessed image
        
        Args:
            image_id: Image ID for visualization
            save_path: Path for saving (optional)
        """
        import matplotlib.pyplot as plt
        

        original = self.load_image(image_id, return_pil=True)
        if original is None:
            print(f"Image {image_id} not found")
            return
        

        preprocessed = self.preprocess_image(image_id, model_type='resnet50')
        
        # Denormalize for visualization
        # ResNet preprocessing: x = (x - mean) / std, reverse it
        # For simplicity, just show resized version
        resized = original.resize(self.target_size, Image.LANCZOS)
        

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original)
        axes[0].set_title(f'Original: {original.size[0]}x{original.size[1]}')
        axes[0].axis('off')
        
        axes[1].imshow(resized)
        axes[1].set_title(f'After resize: {self.target_size[0]}x{self.target_size[1]}')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()

def test_image_loader():
    """
    Tests the ImageLoader
    """
    print("="*70)
    print("STEP 3: Testing ImageLoader")
    print("="*70)

    loader = ImageLoader(target_size=(224, 224))

    test_ids = [15970, 39386, 59263, 21379, 53759]

    print("\n1. Test loading single image:")
    img = loader.load_image(test_ids[0])
    if img is not None:
        print(f"   ✓ Image {test_ids[0]} loaded: shape={img.shape}, dtype={img.dtype}")
    
    print("\n2. Test preprocessing:")
    preprocessed = loader.preprocess_image(test_ids[0], model_type='resnet50')
    if preprocessed is not None:
        print(f"   ✓ Preprocessing done: shape={preprocessed.shape}, dtype={preprocessed.dtype}")
        print(f"   ✓ Value range: [{preprocessed.min():.2f}, {preprocessed.max():.2f}]")
    
    print("\n3. Test loading batch:")
    batch, valid_ids = loader.load_batch(test_ids, model_type='resnet50')
    print(f"   ✓ Batch loaded: shape={batch.shape}")
    print(f"   ✓ Successfully loaded: {len(valid_ids)}/{len(test_ids)} images")
    print(f"   ✓ Valid IDs: {valid_ids}")
    
    print("\n4. Preprocessing visualization:")
    loader.visualize_preprocessing(test_ids[0], save_path='preprocessing_example.png')
    
    print("\n5. Test handling non-existent image:")
    fake_img = loader.load_image(999999)
    if fake_img is None:
        print("   ✓ Correctly handled non-existent ID")
    
    print("\n" + "="*70)
    print("Testing completed!")
    print("="*70)


if __name__ == "__main__":
    test_image_loader()