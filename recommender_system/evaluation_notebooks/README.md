# ML Recommender Evaluation Notebooks

This directory contains comprehensive evaluation notebooks for analyzing the ML wardrobe recommendation system. Each notebook addresses a specific research question about the system's performance and behavior.

## Notebooks Overview

### 1. `evaluation_01_color_bias_analysis.ipynb`
**Research Question**: Does the model recommend clothes only of similar color?

Analyzes whether the recommendation system is biased toward matching colors, or if it considers other factors.

**Key Metrics**:
- Color match rate
- Color diversity in recommendations
- Similarity scores for matching vs non-matching colors

### 2. `evaluation_02_pattern_texture_analysis.ipynb`
**Research Question**: Does it ignore patterns or textures?

Evaluates whether the model preserves pattern information (striped, printed, solid) in recommendations.

**Key Metrics**:
- Pattern preservation rate
- Pattern diversity
- Pattern transition matrix

### 3. `evaluation_03_clothing_type_performance.ipynb`
**Research Question**: Does it fail on certain clothing types?

Tests recommendation quality across different article types (Tshirts, Jeans, Shoes, etc.).

**Key Metrics**:
- Type-specific similarity scores
- Category match rate per type
- Recommendation diversity per type

### 4. `evaluation_04_clustering_quality.ipynb`
**Research Question**: Does it cluster outfits correctly based on embeddings?

Analyzes the clustering structure of the embedding space to see if similar items cluster together.

**Key Metrics**:
- Silhouette score
- Davies-Bouldin index
- Cluster purity (by category/color)
- Intra vs inter-cluster distances

### 5. `evaluation_05_repetitiveness_analysis.ipynb`
**Research Question**: Are recommendations too repetitive?

Measures diversity and overlap in recommendations across different queries.

**Key Metrics**:
- Recommendation diversity (item, category, color)
- Jaccard similarity between recommendation sets
- Overlap matrix

### 6. `evaluation_06_embedding_space_analysis.ipynb`
**Research Question**: Is the embedding space meaningful (clear clusters)?

Comprehensive analysis of the embedding space structure and quality.

**Key Metrics**:
- PCA explained variance
- Distance distribution
- Attribute-based clustering (category, color, gender)
- Embedding space visualizations

## Usage

### Prerequisites

1. Ensure all dependencies are installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn faiss-cpu jupyter
```

2. Make sure the following files exist:
   - `feature_extraction/resnet50_features_pca512.npy` (or FAISS index)
   - `feature_extraction/resnet50_metadata.csv`
   - `recommender_system/faiss_index/product_index.faiss`

### Running the Notebooks

1. Start Jupyter:
```bash
cd recommender_system/evaluation_notebooks
jupyter notebook
```

2. Open any notebook and run all cells (Cell → Run All)

3. Results will be saved to:
   - Plots: `evaluation_results/plots/`
   - Metrics: `evaluation_results/metrics/`

### Running from Command Line

You can also run notebooks programmatically:
```bash
jupyter nbconvert --to notebook --execute evaluation_01_color_bias_analysis.ipynb
```

## Notebook Structure

Each notebook follows this structure:

1. **Introduction**: Research question and hypothesis
2. **Setup and Imports**: Load libraries and utilities
3. **Load Data**: Load vectors and metadata from vector DB
4. **Initialize Recommender**: Set up similarity matcher
5. **Generate Test Cases**: Create synthetic test queries
6. **Analysis**: Run evaluations and compute metrics
7. **Visualizations**: Create insightful graphs
8. **Summary Statistics**: Key findings
9. **Save Results**: Export metrics and plots
10. **Conclusion**: Answer research question with findings

## Shared Utilities

The `evaluation_utils.py` module provides common functions:

- `load_vectors_and_metadata()`: Load data from vector DB
- `generate_test_cases_by_color()`: Create color-based test cases
- `generate_test_cases_by_type()`: Create type-based test cases
- `compute_color_match_rate()`: Calculate color matching
- `compute_diversity_score()`: Measure diversity
- `compute_cluster_metrics()`: Clustering quality metrics
- `reduce_dimensions()`: Dimensionality reduction for visualization
- `save_plot()`: Save plots to results directory
- `save_metrics()`: Save metrics to JSON

## Output Files

### Plots
All plots are saved to `evaluation_results/plots/`:
- `color_bias_*.png`
- `pattern_*.png`
- `clothing_type_*.png`
- `clustering_*.png`
- `repetitiveness_*.png`
- `embedding_space_*.png`

### Metrics
All metrics are saved to `evaluation_results/metrics/`:
- `*_metrics.json`: Summary metrics
- `*_detailed.csv`: Detailed results

## Notes

- **Sampling**: Some analyses use sampling for large datasets to improve performance
- **Synthetic Test Cases**: Since we don't have ground truth, test cases are generated from the dataset
- **Honest Reporting**: Notebooks are designed to report findings honestly, even if results are weak
- **Limitations**: Each notebook documents its limitations

## Troubleshooting

**Import Errors**:
- Make sure you're running from the `evaluation_notebooks` directory
- Check that `evaluation_utils.py` is in the same directory

**Memory Issues**:
- Reduce sample sizes in notebooks
- Use FAISS index instead of loading full .npy file

**Missing Data**:
- Ensure vector DB is initialized
- Check that metadata file exists

## Next Steps

After running all notebooks:

1. Review the generated plots and metrics
2. Fill in the conclusion sections with your findings
3. Document any limitations or issues discovered
4. Use insights to improve the recommendation system

