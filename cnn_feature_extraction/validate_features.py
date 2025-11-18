import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import json


class FeatureValidator:
    """
    Class for validating quality of extracted features
    """

    def __init__(self, features_dir: str = 'extracted_features', model_name: str = 'resnet50'):
        """
        Initialization
        
        Args:
            features_dir: Folder with extracted features
            model_name: Model name
        """

        self.features_dir = Path(features_dir)
        self.model_name = model_name
        self.output_dir = self.features_dir / 'validation'
        self.output_dir.mkdir(exist_ok=True)

        print("="*70)
        print("STEP 8: Validating feature quality")
        print("="*70)

        print("\n1. Loading data...")
        self.features_normalized = self._load_features('normalized')
        self.metadata = self._load_metadata()
        
        print(f"   ✓ Features: {self.features_normalized.shape}")
        print(f"   ✓ Metadata: {len(self.metadata)}")

    def _load_features(self, feature_type: str = 'normalized') -> np.ndarray:
        """Loads feature vectors"""
        if feature_type == 'normalized':
            path = self.features_dir / f'{self.model_name}_features_normalized.npy'
        else:
            path = self.features_dir / f'{self.model_name}_features.npy'
        
        return np.load(path)

    def _load_metadata(self) -> pd.DataFrame:
        """Loads metadata"""
        path = self.features_dir / f'{self.model_name}_metadata.csv'
        return pd.read_csv(path)

    def compute_category_similarities(self, category_column: str = 'articleType'):
        """
        Computes intra- and inter-category similarity
        
        Args:
            category_column: Column with categories
        """
        print(f"\n2. Analyzing similarity by categories ({category_column})...")
        
        categories = self.metadata[category_column].unique()
        print(f"   Categories found: {len(categories)}")

        top_categories = self.metadata[category_column].value_counts().head(10).index.tolist()

        intra_similarities = {}
        inter_similarities = {}

        for category in top_categories:
            cat_indices = self.metadata[self.metadata[category_column] == category_column].index.tolist()

            if len(cat_indices) < 2:
                continue

            cat_features = self.features_normalized[cat_indices]

            sim_matrix = cosine_similarity(cat_features)

            np.fill_diagonal(sim_matrix, 0)
            intra_sim = sim_matrix[sim_matrix > 0].mean()
            intra_similarities[category] = intra_sim

            other_indices = self.metadata[self.metadata[category_column] != category].index.tolist()[:100]
            other_features = self.features_normalized[other_indices]
            inter_sim_matrix = cosine_similarity(cat_features, other_features)
            inter_sim = inter_sim_matrix.mean()
            inter_similarities[category] = inter_sim

        print(f"\n   Results for top-10 categories:")
        print(f"   {'Category':<25} {'Intra-sim':<12} {'Inter-sim':<12} {'Difference':<12}")
        print("   " + "-"*60)

        for category in top_categories[:10]:
            if category in intra_similarities:
                intra = intra_similarities[category]
                inter = inter_similarities[category]
                diff = intra - inter
                print(f"   {category:<25} {intra:<12.4f} {inter:<12.4f} {diff:<12.4f}")

        avg_intra = np.mean(list(intra_similarities.values()))
        avg_inter = np.mean(list(inter_similarities.values()))
        
        print(f"\n   Average:")
        print(f"   Intra-category similarity: {avg_intra:.4f}")
        print(f"   Inter-category similarity: {avg_inter:.4f}")
        print(f"   Difference: {avg_intra - avg_inter:.4f}")
        
        if avg_intra > avg_inter:
            print(f"   ✓ Good! Intra > Inter (features distinguish categories)")
        else:
            print(f"   ⚠️  Warning! Intra <= Inter (features may not be distinguishing enough)")
        

        results = {
            'intra_similarities': {k: float(v) for k, v in intra_similarities.items()},
            'inter_similarities': {k: float(v) for k, v in inter_similarities.items()},
            'avg_intra': float(avg_intra),
            'avg_inter': float(avg_inter),
            'difference': float(avg_intra - avg_inter)
        }
        
        results_path = self.output_dir / f'{category_column}_similarity_analysis.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   ✓ Results saved: {results_path}")
        
        return results

    def visualize_tsne(self, category_column: str = 'masterCategory', n_samples: int = 2000):
        """
        Visualizes features using t-SNE
        
        Args:
            category_column: Column for coloring points
            n_samples: Number of samples for visualization
        """
        print(f"\n3. t-SNE visualization (n_samples={n_samples})...")
        
        if len(self.features_normalized) > n_samples:
            indices = np.random.choice(len(self.features_normalized), n_samples, replace=False)
            features_sample = self.features_normalized[indices]
            metadata_sample = self.metadata.iloc[indices]
        else:
            features_sample = self.features_normalized
            metadata_sample = self.metadata
        
        print(f"   Applying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features_sample)
        
        print(f"   Creating visualization...")
        plt.figure(figsize=(12, 10))
        
        categories = metadata_sample[category_column].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
        
        for i, category in enumerate(categories):
            mask = metadata_sample[category_column] == category
            plt.scatter(
                features_2d[mask, 0], 
                features_2d[mask, 1],
                c=[colors[i]], 
                label=category, 
                alpha=0.6,
                s=20
            )
        
        plt.title(f't-SNE Visualization of Features ({category_column})', fontsize=14)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        
        output_path = self.output_dir / f'tsne_{category_column}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Visualization saved: {output_path}")
        plt.close()

    def find_similar_examples(self, n_examples: int = 5):
        """
        Finds and saves examples of similar images
        
        Args:
            n_examples: Number of examples
        """
        print(f"\n4. Finding examples of similar images...")
        
        examples = []
        
        for _ in range(n_examples):
            query_idx = np.random.randint(0, len(self.features_normalized))
            query_feature = self.features_normalized[query_idx:query_idx+1]
            
            similarities = cosine_similarity(query_feature, self.features_normalized)[0]
            top_indices = np.argsort(similarities)[::-1][1:6] 
            
            query_info = self.metadata.iloc[query_idx]
            similar_info = self.metadata.iloc[top_indices]
            
            example = {
                'query_id': int(query_info['id']),
                'query_category': query_info['articleType'],
                'similar': [
                    {
                        'id': int(row['id']),
                        'category': row['articleType'],
                        'similarity': float(similarities[idx])
                    }
                    for idx, row in zip(top_indices, similar_info.to_dict('records'))
                ]
            }
            examples.append(example)
            
            print(f"   Query {_+1}: ID={query_info['id']}, Category={query_info['articleType']}")
            for sim in example['similar'][:3]:
                print(f"      → ID={sim['id']}, Category={sim['category']}, Sim={sim['similarity']:.4f}")
        
        examples_path = self.output_dir / 'similar_examples.json'
        with open(examples_path, 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"   ✓ Examples saved: {examples_path}")


def main():
    """
    Main function for validation
    """
    validator = FeatureValidator(
        features_dir='extracted_features',
        model_name='resnet50'
    )
    
    validator.compute_category_similarities(category_column='articleType')
    validator.compute_category_similarities(category_column='masterCategory')
    
    validator.visualize_tsne(category_column='masterCategory', n_samples=2000)
    validator.visualize_tsne(category_column='gender', n_samples=2000)
    
    validator.find_similar_examples(n_examples=5)
    
    print("\n" + "="*70)
    print("Validation completed!")
    print("="*70)
    print(f"\nResults saved in: {validator.output_dir}")


if __name__ == "__main__":
    main()