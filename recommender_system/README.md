# Recommender System

This folder contains the core recommendation system components.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the GUI application
python wardrobe_recommender_app.py

# Or test the recommender
python test_recommender.py ../Wardrobe_upload_system/wardrobe_storage/aasritha
```

## Components

- `wardrobe_recommender_app.py` - Main GUI application
- `recommender.py` - Recommendation engine
- `similarity.py` - Similarity matching
- `feature_extractor.py` - Feature extraction
- `preprocessing.py` - Image preprocessing

## Documentation

📚 **Complete documentation is available in the [docs folder](../docs/recommender_system/):**

- [Application Guide](../docs/recommender_system/app_guide.md) - Complete GUI application documentation
- [Usage Guide](../docs/recommender_system/usage_guide.md) - How to use the recommendation system
- [Components](../docs/recommender_system/components.md) - Core system components overview
- [Recommendation Strategies](../docs/recommender_system/recommendation_strategies.md) - Detailed strategy analysis

## Requirements

See `requirements.txt` for dependencies.

## FAISS Vector Database

The recommender system uses FAISS (Facebook AI Similarity Search) for efficient similarity search and feature storage.

### Features

- **Automatic Initialization**: The system automatically checks if a FAISS index exists at startup
  - If index exists → Loads it (fast startup)
  - If index doesn't exist but `.npy` file exists → Automatically creates index from `.npy` file
  - If neither exists → Raises error with helpful message
- **User Feature Caching**: User wardrobe features are cached in the vector DB for faster subsequent recommendations
- **Efficient Search**: FAISS provides faster similarity search compared to in-memory NumPy operations
- **Persistent Storage**: Features are stored on disk, reducing memory usage

### Storage Structure

```
recommender_system/
├── faiss_index/
│   ├── product_index.faiss      # FAISS index for product features
│   ├── product_ids.pkl          # Mapping of index positions to product IDs
│   └── user_features/            # User wardrobe features
│       ├── {user_id}_features.npy
│       └── {user_id}_metadata.json
```

### Usage

The FAISS vector database is automatically initialized when you create a `Recommender` or `SimilarityMatcher` instance:

```python
from recommender import Recommender

# FAISS index will be auto-created on first run if .npy file exists
recommender = Recommender()
```

### Manual Migration

If you want to manually migrate features to FAISS (optional):

```bash
python migrate_to_faiss.py
```

Or with custom paths:

```bash
python migrate_to_faiss.py --npy path/to/features.npy --metadata path/to/metadata.csv
```

### Performance Benefits

- **Faster Search**: FAISS uses optimized algorithms for similarity search
- **Reduced Memory**: Index can be memory-mapped, reducing RAM usage
- **Scalability**: Better performance as dataset grows
- **Caching**: User features are cached, avoiding redundant extractions

### Troubleshooting

**Q: FAISS index not found error**
- Ensure `resnet50_features_pca512.npy` exists in `extracted_features/` directory
- The index will be created automatically on first run

**Q: How to rebuild the index?**
- Delete `faiss_index/product_index.faiss` and `faiss_index/product_ids.pkl`
- Restart the application - index will be recreated automatically

**Q: User features not being cached?**
- Ensure `user_id` is provided when calling `get_recommendations()`
- Check that `faiss_index/user_features/` directory is writable

### Backward Compatibility

The system maintains backward compatibility:
- If FAISS is not available or index creation fails, it falls back to NumPy-based similarity
- Both methods produce the same results (within numerical precision)
