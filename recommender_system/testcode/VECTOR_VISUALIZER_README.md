# Vector Database 3D Visualizer

A simple Python-based 3D scatter plot visualization for the FAISS vector database.

## Features

- **3D Scatter Plot**: Visualize vectors in 3D space using PCA or t-SNE
- **Interactive**: Click on points to see product information
- **Color Coding**: Color points by category, gender, article type, or color
- **Sampling**: Automatically samples large datasets for faster visualization
- **Simple GUI**: Uses matplotlib's interactive 3D plot

## Usage

### Basic Usage

```bash
cd recommender_system
python vector_visualizer.py
```

Or from project root:

```bash
python recommender_system/vector_visualizer.py
```

### With Virtual Environment

```bash
source venv/bin/activate
python recommender_system/vector_visualizer.py
```

## How It Works

1. **Loads Vectors**: 
   - First tries to load from `resnet50_features_pca512.npy` (fastest)
   - Falls back to reconstructing from FAISS index if needed

2. **Dimension Reduction**:
   - **PCA**: Fast, linear reduction (default for large datasets)
   - **t-SNE**: Slower but better visualization of clusters

3. **Visualization**:
   - Creates interactive 3D scatter plot
   - Click on any point to see product details
   - Rotate, zoom, and pan the 3D view

## Options

When you run the script, you'll be prompted for:

1. **Reduction Method**:
   - **PCA** (faster): Good for large datasets, linear projection
   - **t-SNE** (slower): Better cluster visualization, non-linear

2. **Sampling**:
   - For datasets > 5000 vectors, you can choose to sample 5000 for faster computation
   - Or use all vectors (slower but complete)

3. **Color By**:
   - `None`: All points same color
   - `masterCategory`: Color by product category
   - `gender`: Color by gender
   - `articleType`: Color by article type
   - `baseColour`: Color by base color

## Requirements

- matplotlib
- scikit-learn (for PCA/t-SNE)
- numpy
- pandas
- faiss-cpu
- tkinter (usually comes with Python)

Install missing dependencies:

```bash
pip install matplotlib scikit-learn
```

## Tips

- **Large Datasets**: Use sampling (5000 vectors) for faster visualization
- **Better Clusters**: Use t-SNE for better visualization of similar products
- **Faster**: Use PCA for quick visualization
- **Interactive**: Click and drag to rotate, scroll to zoom, right-click to pan

## Troubleshooting

**Error: "No module named 'matplotlib'"**
```bash
pip install matplotlib
```

**Error: "No module named 'sklearn'"**
```bash
pip install scikit-learn
```

**t-SNE is very slow**
- Use PCA instead, or enable sampling

**Can't see the plot**
- Make sure you have a display/GUI available
- On Linux servers, you may need X11 forwarding or use a different backend

## Example

```python
from vector_visualizer import VectorVisualizer

# Create visualizer
viz = VectorVisualizer()

# Reduce dimensions with PCA
viz.reduce_dimensions(method='PCA', sample_size=5000)

# Visualize with color coding
viz.visualize(color_by='masterCategory')
```

