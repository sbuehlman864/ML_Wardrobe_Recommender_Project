import kagglehub
import pandas as pd

def load_data():
    # Download latest version
    path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")

    print("Path to dataset files:", path)

    csv_file = f"{path}/fashion-dataset/styles.csv"
    image_file = f"{path}/fashion-dataset/images.csv"
    try:
        data = pd.read_csv(csv_file, error_bad_lines=False)
        images = pd.read_csv(image_file, error_bad_lines=False)
    except pd.errors.ParserError as e:
        print("Error parsing CSV files:", e)
        data = pd.DataFrame()
        images = pd.DataFrame()

    print("Data loaded successfully!")
    print(data.head())
    print(images.head())
    
    return data, images