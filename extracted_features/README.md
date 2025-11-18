# Extracted Features

This folder contains pre-extracted CNN features from 41,802 fashion images using ResNet50.

## 📁 Available Files

### ✅ Files in this repository (uploaded):
- **`resnet50_features_pca512.npy`** (82 MB) - PCA-reduced features (512 dimensions)
- **`resnet50_metadata.csv`** (3.9 MB) - Image metadata with IDs and categories
- **`resnet50_pca512_model.pkl`** (4 MB) - PCA model for transforming new images
- **`resnet50_pca512_info.json`** (14 KB) - PCA statistics and variance info
- **`resnet50_extraction_info.json`** (391 B) - Extraction process statistics
- **`resnet50_failed_ids.txt`** (30 B) - List of failed image IDs
- **`validation/`** - Validation results and visualizations

### ❌ Large files NOT in repository (>100MB):
Due to GitHub file size limits, these files are not included:
- **`resnet50_features.npy`** (327 MB) - Original features (2048 dimensions)
- **`resnet50_features_normalized.npy`** (327 MB) - L2-normalized features (2048 dimensions)

## 📥 How to Get Large Files

Contact the project team to obtain large files via:
- Google Drive
- Dropbox
- OneDrive
- Or other file sharing services

## 🚀 Usage

For most use cases, the **PCA version (512 dimensions)** is sufficient and much faster:

```python
import numpy as np
import pandas as pd

# Load PCA features (included in repo)
features = np.load('extracted_features/resnet50_features_pca512.npy')
metadata = pd.read_csv('extracted_features/resnet50_metadata.csv')

print(f"Features shape: {features.shape}")  # (41802, 512)
print(f"Total images: {len(metadata)}")
```

## 📊 Feature Statistics

- **Total images processed:** 41,802
- **Feature dimensions:** 
  - Original: 2048
  - PCA: 512 (94.25% variance retained)
- **Success rate:** 99.99%
- **Model:** ResNet50 (pre-trained on ImageNet)

For more details, see the main README.md in the project root.

