import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from load_data import load_data
from clean_data import clean_data
from image_loader import ImageLoader
from cnn_model import FeatureExtractor

class DatasetFeatureExtractor:
    """
    Class for extracting features from the entire dataset
    """

    def __init__(
        self,
        model_name: str = 'resnet50',
        batch_size: int = 32,
        output_dir: str = 'extracted_features'
    ):
        """
        Initialization
        
        Args:
            model_name: Model name for feature extraction
            batch_size: Batch size for processing
            output_dir: Folder for saving results
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("="*70)
        print("STEP 5: Extracting features from entire dataset")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  - Model: {model_name}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Output directory: {output_dir}")
        
        print("\n1. Loading and cleaning data...")
        data, images, dataset_path = load_data()
        self.data, self.images = clean_data(data, images)
        print(f"   ✓ Records loaded: {len(self.data)}")
        
        print("\n2. Initializing components...")
        self.image_loader = ImageLoader(target_size=(224, 224))
        self.feature_extractor = FeatureExtractor(model_name=model_name)
        
        print("\n✓ Ready to extract features!")

    def extract_all_features(self, save_every: int = 100):
        """
        Extracts features from all images in the dataset
        
        Args:
            save_every: Save intermediate results every N batches
        """
        print("\n" + "="*70)
        print("3. Extracting features from all images")
        print("="*70)

        all_ids = self.data['id'].tolist()
        total_images = len(all_ids)

        all_features = []
        valid_ids = []
        failed_ids = []

        num_batches = (total_images + self.batch_size - 1) // self.batch_size

        print(f"\nTotal images: {total_images}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches: {num_batches}")
        print(f"\nStarting processing...\n")

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, total_images)
            batch_ids = all_ids[start_idx:end_idx]

            batch_images, batch_valid_ids = self.image_loader.load_batch(
                batch_ids,
                model_type=self.model_name
            )

            batch_failed = [img_id for img_id in batch_ids if img_id not in batch_valid_ids]
            failed_ids.extend(batch_failed)

            if len(batch_valid_ids) > 0:
                features = self.feature_extractor.extract_features(batch_images)
                all_features.append(features)
                valid_ids.extend(batch_valid_ids)

            if (batch_idx + 1) % save_every == 0:
                self._save_intermediate_results(
                    all_features, valid_ids, failed_ids, batch_idx + 1
                )
        
        print(f"\n✓ Processing completed!")
        print(f"   Successful: {len(valid_ids)}")
        print(f"   Failed: {len(failed_ids)}")
        
        print("\n4. Combining results...")
        features_array = np.vstack(all_features)
        print(f"   ✓ Final array shape: {features_array.shape}")
        
        print("\n5. Saving results...")
        self._save_final_results(features_array, valid_ids, failed_ids)
        
        return features_array, valid_ids, failed_ids

    def _save_intermediate_results(self, features_list, valid_ids, failed_ids, batch_num):
        """Saves intermediate results"""
        if features_list:
            features_array = np.vstack(features_list)
            temp_path = self.output_dir / f'temp_features_batch_{batch_num}.npy'
            np.save(temp_path, features_array)

    def _save_final_results(self, features_array, valid_ids, failed_ids):
        """
        Saves final results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        features_path = self.output_dir / f"{self.model_name}_features.npy"
        np.save(features_path, features_array)
        print(f"   ✓ Features saved: {features_path}")

        metadata_df = self.data[self.data['id'].isin(valid_ids)].copy()
        metadata_df = metadata_df.set_index('id').loc[valid_ids].reset_index()
        metadata_path = self.output_dir / f'{self.model_name}_metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)
        print(f"   ✓ Metadata saved: {metadata_path}")

        if failed_ids:
            failed_path = self.output_dir / f'{self.model_name}_failed_ids.txt'
            with open(failed_path, 'w') as f:
                for img_id in failed_ids:
                    f.write(f"{img_id}\n")
            print(f"   ✓ Failed IDs saved: {failed_path}")
        

        info = {
            'model_name': self.model_name,
            'feature_dimension': features_array.shape[1],
            'total_images_processed': len(valid_ids),
            'failed_images': len(failed_ids),
            'success_rate': len(valid_ids) / (len(valid_ids) + len(failed_ids)) * 100,
            'batch_size': self.batch_size,
            'timestamp': timestamp,
            'features_shape': list(features_array.shape),
            'features_stats': {
                'mean': float(features_array.mean()),
                'std': float(features_array.std()),
                'min': float(features_array.min()),
                'max': float(features_array.max())
            }
        }

        info_path = self.output_dir / f'{self.model_name}_extraction_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"   ✓ Extraction info saved: {info_path}")

        temp_files = list(self.output_dir.glob('temp_features_*.npy'))
        for temp_file in temp_files:
            temp_file.unlink()
        if temp_files:
            print(f"   ✓ Removed {len(temp_files)} temporary files")
        
        print(f"\n" + "="*70)
        print("All results saved!")
        print("="*70)
        print(f"\nFiles:")
        print(f"  1. {features_path.name} - feature vectors ({features_array.shape})")
        print(f"  2. {metadata_path.name} - image metadata")
        print(f"  3. {info_path.name} - extraction information")
        if failed_ids:
            print(f"  4. {failed_path.name} - list of failed IDs ({len(failed_ids)} pcs.)")


def main():
    """
    Main function for running feature extraction
    """

    extractor = DatasetFeatureExtractor(
        model_name='resnet50',
        batch_size=32,  # Reduce to 16 if running out of memory
        output_dir='extracted_features'
    )
    
    # Extract features
    features, valid_ids, failed_ids = extractor.extract_all_features(save_every=50)
    
    print("\n" + "="*70)
    print("SUCCESSFULLY COMPLETED!")
    print("="*70)
    print(f"\nStatistics:")
    print(f"  - Processed images: {len(valid_ids)}")
    print(f"  - Feature dimension: {features.shape[1]}")
    print(f"  - Failed attempts: {len(failed_ids)}")
    print(f"  - Success rate: {len(valid_ids)/(len(valid_ids)+len(failed_ids))*100:.2f}%")
    
    print(f"\nNext steps:")
    print(f"  1. Feature vectors normalization (L2 normalization)")
    print(f"  2. Optionally: PCA for dimensionality reduction")
    print(f"  3. Feature quality validation")
    print(f"  4. Pass results to team for KNN")


if __name__ == "__main__":
    main()