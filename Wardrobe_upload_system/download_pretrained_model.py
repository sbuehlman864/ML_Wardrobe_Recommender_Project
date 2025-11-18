"""
Download Pre-trained Clothing Detection YOLO Model
Downloads from Hugging Face or other sources
"""

import os
from ultralytics import YOLO
import subprocess
import sys

def download_huggingface_model():
    """Download pre-trained clothing detection model from Hugging Face"""
    print("Downloading pre-trained clothing detection model from Hugging Face...")
    
    try:
        # Try using ultralyticsplus for Hugging Face models
        try:
            import ultralyticsplus
            from ultralyticsplus import YOLO as YOLOPlus
            
            print("Using ultralyticsplus to load Hugging Face model...")
            model = YOLOPlus('kesimeg/yolov8n-clothing-detection')
            
            # Save model locally
            model_path = 'yolov8n-clothing-detection.pt'
            model.model.save(model_path)
            print(f"✓ Model saved to: {model_path}")
            return model_path
            
        except ImportError:
            print("ultralyticsplus not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralyticsplus"])
            
            from ultralyticsplus import YOLO as YOLOPlus
            model = YOLOPlus('kesimeg/yolov8n-clothing-detection')
            model_path = 'yolov8n-clothing-detection.pt'
            model.model.save(model_path)
            print(f"✓ Model saved to: {model_path}")
            return model_path
            
    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")
        print("\nTrying alternative method...")
        return None

def download_alternative_model():
    """Try downloading alternative pre-trained model"""
    print("Attempting to download alternative model...")
    
    # Try direct download from Hugging Face
    try:
        import requests
        
        # Hugging Face model URL (this is a placeholder - actual URL may vary)
        # For now, we'll use the ultralytics hub
        print("Using YOLOv8 with custom weights...")
        
        # Alternative: Use a publicly available clothing detection model
        # This would require the actual model URL
        return None
        
    except Exception as e:
        print(f"Alternative download failed: {e}")
        return None

def main():
    """Main download function"""
    print("=" * 60)
    print("Downloading Pre-trained Clothing Detection Model")
    print("=" * 60)
    
    # Try Hugging Face first
    model_path = download_huggingface_model()
    
    if model_path and os.path.exists(model_path):
        print("\n" + "=" * 60)
        print("✓ Success! Model downloaded successfully")
        print(f"Model saved to: {model_path}")
        print("=" * 60)
        print("\nYou can now use this model in wardrobe_app.py")
        return model_path
    else:
        print("\n" + "=" * 60)
        print("Manual Download Required")
        print("=" * 60)
        print("\nPlease download a pre-trained clothing detection model:")
        print("\nOption 1: Hugging Face Model")
        print("  1. Install: pip install ultralyticsplus")
        print("  2. In Python:")
        print("     from ultralyticsplus import YOLO")
        print("     model = YOLO('kesimeg/yolov8n-clothing-detection')")
        print("     model.model.save('yolov8n-clothing-detection.pt')")
        print("\nOption 2: Use ModaNet YOLOv3")
        print("  Download from: https://github.com/kritanjalijain/Clothing_Detection_YOLO")
        print("\nOption 3: Use standard YOLO with workaround")
        print("  The app will use person detection + region extraction")
        return None

if __name__ == "__main__":
    main()

