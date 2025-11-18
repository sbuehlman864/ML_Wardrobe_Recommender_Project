"""
Alternative method to load Hugging Face model without ultralyticsplus
Uses huggingface_hub directly to download and load the model
"""

import os
import torch
from ultralytics import YOLO
import requests
from pathlib import Path

def download_model_from_huggingface():
    """Download YOLO model from Hugging Face using direct download"""
    
    # Hugging Face model repository
    repo_id = "kesimeg/yolov8n-clothing-detection"
    
    print("Downloading model from Hugging Face...")
    print(f"Repository: {repo_id}")
    
    # Try using huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
        
        print("Using huggingface_hub...")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="yolov8n-clothing-detection.pt",
            local_dir="./",
            local_dir_use_symlinks=False
        )
        
        if os.path.exists(model_path):
            print(f"✓ Model downloaded to: {model_path}")
            return model_path
        else:
            # Try alternative filename
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.pt",
                local_dir="./",
                local_dir_use_symlinks=False
            )
            return model_path
            
    except ImportError:
        print("huggingface_hub not installed. Installing...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="yolov8n-clothing-detection.pt",
            local_dir="./",
            local_dir_use_symlinks=False
        )
        return model_path

def download_model_direct():
    """Direct download using requests (fallback)"""
    print("Attempting direct download...")
    
    # Try to get model file directly
    # Note: This URL structure may vary
    base_url = "https://huggingface.co/kesimeg/yolov8n-clothing-detection/resolve/main"
    
    filenames = [
        "best.pt",  # Most common name in Hugging Face repos
        "yolov8n-clothing-detection.pt",
        "model.pt",
        "weights.pt"
    ]
    
    for filename in filenames:
        url = f"{base_url}/{filename}"
        try:
            print(f"Trying: {url}")
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                model_path = f"./{filename}"
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✓ Model downloaded to: {model_path}")
                
                # Rename to standard name for consistency
                standard_name = "yolov8n-clothing-detection.pt"
                if filename != standard_name and not os.path.exists(standard_name):
                    try:
                        import shutil
                        shutil.copy2(model_path, standard_name)
                        print(f"✓ Also saved as: {standard_name}")
                    except:
                        pass
                
                return model_path
        except Exception as e:
            print(f"Failed: {e}")
            continue
    
    return None

def load_model_alternative():
    """Load model using alternative method"""
    model_path = None
    
    # Method 1: Try huggingface_hub
    try:
        model_path = download_model_from_huggingface()
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Try direct download
    if not model_path or not os.path.exists(model_path):
        try:
            model_path = download_model_direct()
        except Exception as e:
            print(f"Method 2 failed: {e}")
    
    # Method 3: Use standard YOLOv8 and provide instructions
    if not model_path or not os.path.exists(model_path):
        print("\n" + "="*60)
        print("Could not download pre-trained model automatically.")
        print("="*60)
        print("\nAlternative options:")
        print("\n1. Use standard YOLOv8 (will detect person, then extract regions):")
        print("   - The app will fall back to person detection")
        print("   - Less accurate but works without pre-trained model")
        print("\n2. Manual download:")
        print("   - Visit: https://huggingface.co/kesimeg/yolov8n-clothing-detection")
        print("   - Download the .pt model file")
        print("   - Save as: yolov8n-clothing-detection.pt")
        print("\n3. Install dependencies separately:")
        print("   pip install huggingface_hub")
        print("   python load_model_alternative.py")
        return None
    
    # Load model with ultralytics
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    model = load_model_alternative()
    if model:
        print("\n✓ Success! Model is ready to use.")
        print("You can now use this model in wardrobe_app.py")
    else:
        print("\n✗ Could not load model. See instructions above.")

