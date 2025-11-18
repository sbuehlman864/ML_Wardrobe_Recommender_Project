import os
import pandas as pd
from PIL import Image
from pathlib import Path
from load_data import load_data
from clean_data import clean_data

def explore_image_structure():
    """
    Explores the structure of the image folder and its relationship with metadata
    """
    print("="*70)
    print("STEP 2: Exploring image data structure")
    print("="*70)

    print("\n1. Loading data...")
    data, images, dataset_path = load_data()
    data, images = clean_data(data, images)
    print(f"   Number of records in cleaned_data: {len(data)}")

    print(f"\n2. Dataset path:\n   {dataset_path}")
    
    # Search for images folder
    images_folder = None
    possible_image_folders = [
        Path(dataset_path) / "fashion-dataset" / "images",
        Path(dataset_path) / "images",
    ]
    
    for folder in possible_image_folders:
        if folder.exists():
            images_folder = folder
            break
    
    # If standard paths not found, search recursively
    if images_folder is None:
        print(f"\n3. Searching for images folder...")
        for root, dirs, files in os.walk(dataset_path):
            img_files = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))]
            if img_files and len(img_files) > 100:  # Should have many files
                print(f"   Images found in: {root}")
                images_folder = Path(root)
                break
    
    if images_folder is None:
        print("   ERROR: Images folder not found!")
        return None, 0, 0
    
    print(f"\n3. Images folder:\n   {images_folder}")
    print(f"   Exists: {images_folder.exists()}")
    
    # Count files
    if images_folder.exists():
        image_files = list(images_folder.glob("*.jpg")) + list(images_folder.glob("*.png"))
        print(f"\n4. Images found in folder: {len(image_files)}")
        
        # Show first few files
        print(f"\n5. Example file names:")
        for img_file in list(image_files)[:5]:
            print(f"   - {img_file.name}")

    print(f"\n6. Mapping IDs to images:")
    sample_ids = data['id'].head(10).tolist()
    print(f"   First 10 IDs from CSV: {sample_ids}")

    found_count = 0
    missing_count = 0
    image_info = []

    for img_id in sample_ids:
        possible_names = [
            f"{img_id}.jpg",
            f"{img_id}.png",
            f"{img_id}.jpeg",
        ]

        found = False
        for name in possible_names:
            img_path = images_folder / name
            if img_path.exists():
                found = True
                found_count += 1

                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        mode = img.mode
                        image_info.append({
                            'id': img_id,
                            'path': str(img_path),
                            'width': width,
                            'height': height,
                            'mode': mode,
                            'exists': True
                        })
                        print(f"   ✓ ID {img_id}: {name} - {width}x{height} ({mode})")
                except Exception as e:
                    print(f"   ✗ ID {img_id}: Opening error - {e}")
                break

        if not found:
            missing_count += 1
            image_info.append({
                'id': img_id,
                'path': None,
                'width': None,
                'height': None,
                'mode': None,
                'exists': False
            })
            print(f"   ✗ ID {img_id}: Image not found")
    
    print(f"\n7. Statistics for first 10 IDs:")
    print(f"   Found: {found_count}")
    print(f"   Missing: {missing_count}")

    print(f"\n8. Full check of all IDs from cleaned_data...")
    all_ids = data['id'].tolist()
    total_found = 0
    total_missing = 0
    all_sizes = []

    for img_id in all_ids:
        img_path = images_folder / f"{img_id}.jpg"
        if img_path.exists():
            total_found += 1
            try:
                with Image.open(img_path) as img:
                    all_sizes.append(img.size)
            except:
                pass
        else:
            total_missing += 1
    
    print(f"   Total IDs in cleaned_data: {len(all_ids)}")
    print(f"   Images found: {total_found}")
    print(f"   Missing: {total_missing}")
    print(f"   Percentage found: {(total_found/len(all_ids)*100):.2f}%")
    
    if all_sizes:
        print(f"\n9. Image size statistics:")
        widths = [size[0] for size in all_sizes]
        heights = [size[1] for size in all_sizes]
        
        print(f"   Width - Min: {min(widths)}, Max: {max(widths)}, Avg: {sum(widths)/len(widths):.1f}")
        print(f"   Height - Min: {min(heights)}, Max: {max(heights)}, Avg: {sum(heights)/len(heights):.1f}")
        
    
        aspect_ratios = [w/h for w, h in all_sizes[:100]]  
        print(f"   Aspect ratio (first 100): {min(aspect_ratios):.2f} - {max(aspect_ratios):.2f}")
    
    print(f"\n10. Saving path information...")
    config = {
        'dataset_path': str(dataset_path),
        'images_folder': str(images_folder),
        'total_images': total_found,
        'csv_path': os.path.join(str(dataset_path), "fashion-dataset", "styles.csv")
    }
    
    import json
    with open('dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   Configuration saved to dataset_config.json")
    
    print("\n" + "="*70)
    print("Exploration completed!")
    print("="*70)
    
    return images_folder, total_found, total_missing

if __name__ == "__main__":
    explore_image_structure()