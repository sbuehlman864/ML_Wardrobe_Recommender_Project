# Recommender System

This folder contains the core recommendation system components.

## Components

### 1. `preprocessing.py` - Image Preprocessing
- Preprocesses images for ResNet50 feature extraction
- Handles image resizing, normalization, quality checks
- Supports batch processing

### 2. `feature_extractor.py` - Feature Extraction
- Extracts 512-dimensional features from user wardrobe images
- Uses ResNet50 + PCA (from extracted_features folder)
- Supports batch processing and folder processing

### 3. `similarity.py` - Similarity Matching (TODO)
- Calculates similarity between user wardrobe and products
- Cosine similarity and other metrics

### 4. `recommender.py` - Recommendation Engine (TODO)
- Generates recommendations based on user wardrobe
- Multiple recommendation strategies

### 5. `database.py` - Database Operations (TODO)
- SQLite database for user wardrobe storage
- CRUD operations for wardrobe items

## Setup

### Install Dependencies

```bash
pip install torch torchvision pillow numpy pandas
```

### Download Dataset

``` #!/bin/bash
curl -L -o ~/Downloads/fashion-product-images-small.zip\
  https://www.kaggle.com/api/v1/datasets/download/paramaggarwal/fashion-product-images-small ```

### Download ResNet50 Model

If you encounter SSL certificate errors, download the model manually:

```bash
./download_resnet50.sh
```

Or fix SSL certificates:

```bash
./fix_ssl_certificates.sh
```

## Usage

### Quick Start - Get Recommendations

**Option 1: Test with your wardrobe folder (Recommended)**
```bash
python test_recommender.py ../Wardrobe_upload_system/wardrobe_storage/jashwanth
```

**Option 2: Test with a single image**
```bash
python quick_test.py ../Wardrobe_upload_system/wardrobe_storage/jashwanth/image.png
```

**Option 3: Use in Python code**
```python
from recommender import Recommender

# Initialize
recommender = Recommender()

# Get recommendations
recommendations = recommender.get_recommendations(
    user_wardrobe_paths=[
        'path/to/image1.jpg',
        'path/to/image2.jpg'
    ],
    strategy='hybrid',  # or 'similar', 'complementary', 'category_expansion'
    top_k=20
)

# View results
print(recommendations[['id', 'articleType', 'baseColour', 'similarity_score']])
```

### Extract Features from Single Image

```bash
python feature_extractor.py path/to/image.jpg
```

### Extract Features from Wardrobe Folder

```python
from feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_from_wardrobe_folder("../Wardrobe_upload_system/wardrobe_storage/jashwanth")
```

### Preprocess Image

```python
from preprocessing import preprocess_image

tensor = preprocess_image("path/to/image.jpg")
```

## Troubleshooting

### SSL Certificate Error

If you get SSL certificate errors when downloading ResNet50:

1. **Quick fix**: Run `./download_resnet50.sh` to download manually
2. **Permanent fix**: Run `./fix_ssl_certificates.sh` or install certificates manually

### Model Not Found

If the model can't be loaded:

1. Check if model exists: `ls ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`
2. If not, run: `./download_resnet50.sh`
3. The code will automatically use the cached model if available

### PCA Model Not Found

Ensure `resnet50_pca512_model.pkl` exists in `../extracted_features/` folder.

## Next Steps

- [ ] Implement similarity matching
- [ ] Build recommendation engine
- [ ] Set up database
- [ ] Create API endpoints
- [ ] Build web interface

