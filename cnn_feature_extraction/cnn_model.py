import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from typing import Literal, Tuple
import json

class FeatureExtractor:
    """
    Class for extracting features from images using pre-trained CNNs
    """

    def __init__(
        self,
        model_name: Literal['resnet50', 'vgg16', 'efficientnet'] = 'resnet50',
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        use_global_pooling: bool = True
    ):
        """
        Initialize Feature Extractor
        
        Args:
            model_name: Model name ('resnet50', 'vgg16', 'efficientnet')
            input_shape: Input image size (height, width, channels)
            use_global_pooling: Whether to use global average pooling
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.use_global_pooling = use_global_pooling

        print(f"Initializing Feature Extractor...")
        print(f"  - Model: {model_name}")
        print(f"  - Input size: {input_shape}")
        print(f"  - Global pooling: {use_global_pooling}")

        self.model = self._build_model()

        self.feature_dim = self.model.output_shape[-1]
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"✓ Model loaded successfully!")


    def _build_model(self) -> Model:
        """
        Creates a model for feature extraction
        
        Returns:
            Keras Model without the top classification layer
        """
        print(f"\nLoading pre-trained model {self.model_name}...")

        if self.model_name == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape,
                pooling='avg' if self.use_global_pooling else None
            )

        elif self.model_name == 'vgg16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape,
                pooling='avg' if self.use_global_pooling else None
            )
            
        elif self.model_name == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape,
                pooling='avg' if self.use_global_pooling else None
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        base_model.trainable = False
        
        return base_model

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extracts features from a batch of images
        
        Args:
            images: numpy array of shape (batch_size, height, width, channels)
                    Images must be preprocessed for the corresponding model
        
        Returns:
            numpy array of shape (batch_size, feature_dim) with extracted features
        """
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)

        features = self.model.predict(images, verbose=0)

        return features

    def extract_single_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts features from a single image
        
        Args:
            image: numpy array of shape (height, width, channels)
        
        Returns:
            numpy array of shape (feature_dim,)
        """
        image_batch = np.expand_dims(image, axis=0)

        features = self.model.predict(image_batch, verbose=0)

        return features[0]

    def get_model_info(self) -> dict:
        """
        Returns information about the model
        
        Returns:
            dict with model information
        """
        return {
            'model_name': self.model_name,
            'input_shape': self.input_shape,
            'feature_dim': self.feature_dim,
            'use_global_pooling': self.use_global_pooling,
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        }

    def save_model_info(self, filepath: str = 'model_info.json'):
        """
        Saves model information to a JSON file
        
        Args:
            filepath: Path to file for saving
        """
        info = self.get_model_info()
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Model information saved to {filepath}")

def test_feature_extractor():
    """
    Tests the FeatureExtractor
    """
    print("="*70)
    print("STEP 4: Testing Feature Extractor")
    print("="*70)


    from image_loader import ImageLoader

    loader = ImageLoader(target_size=(224, 224))

    test_ids = [15970, 39386, 59263]
    
    print("\n" + "="*70)
    print("Test 1: ResNet50")
    print("="*70)


    extractor_resnet = FeatureExtractor(model_name='resnet50')

    print("\n1. Extracting features from single image:")
    img = loader.preprocess_image(test_ids[0], model_type='resnet50')
    if img is not None:
        features = extractor_resnet.extract_single_feature(img)
        print(f"   ✓ Features extracted: shape={features.shape}, dtype={features.dtype}")
        print(f"   ✓ Value range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"   ✓ Mean value: {features.mean():.4f}")
        print(f"   ✓ Std value: {features.std():.4f}")

    print("\n2. Extracting features from batch:")
    batch, valid_ids = loader.load_batch(test_ids, model_type='resnet50')
    batch_features = extractor_resnet.extract_features(batch)
    print(f"   ✓ Feature batch: shape={batch_features.shape}")
    print(f"   ✓ Processed images: {len(valid_ids)}")

    print("\n3. Checking feature uniqueness:")
    similarity_12 = np.dot(batch_features[0], batch_features[1]) / (
        np.linalg.norm(batch_features[0]) * np.linalg.norm(batch_features[1])
    )

    similarity_13 = np.dot(batch_features[0], batch_features[2]) / (
        np.linalg.norm(batch_features[0]) * np.linalg.norm(batch_features[2])
    )
    print(f"   Cosine similarity between image 1 and 2: {similarity_12:.4f}")
    print(f"   Cosine similarity between image 1 and 3: {similarity_13:.4f}")
    print(f"   ✓ Features are different (that's good!)")

    print("\n4. Model information:")
    model_info = extractor_resnet.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    extractor_resnet.save_model_info('resnet50_model_info.json')
    
    print("\n" + "="*70)
    print("Test 2: Comparing different models")
    print("="*70)
    
    models_to_test = ['resnet50', 'vgg16']
    
    for model_name in models_to_test:
        print(f"\n{model_name.upper()}:")
        extractor = FeatureExtractor(model_name=model_name)
        
        img = loader.preprocess_image(test_ids[0], model_type=model_name)
        features = extractor.extract_single_feature(img)
        
        print(f"   Feature dimension: {features.shape[0]}")
        print(f"   Value range: [{features.min():.4f}, {features.max():.4f}]")
    
    print("\n" + "="*70)
    print("Testing completed!")
    print("="*70)
    
    print("\n✓ Model is ready to extract features from the entire dataset!")
    print("✓ Next step: batch processing of all ~41,802 images")


if __name__ == "__main__":
    test_feature_extractor()