import pandas as pd

def clean_data(data: pd.DataFrame, images: pd.DataFrame):
    """
    Cleans the data by removing rows with null values and their corresponding images.

    Parameters:
        data (pd.DataFrame): The main dataframe containing data.
        images (pd.DataFrame): The dataframe containing image data.

    Returns:
        (pd.DataFrame, pd.DataFrame): The cleaned data and images dataframes.
    """
    # Remove rows with the masterCategory of Home, Sporting Goods, Free Items, and Personal Care
    data = data[~data['masterCategory'].isin(['Home', 'Sporting Goods', 'Free Items', 'Personal Care'])]

    # Find rows with no null values in the data dataframe
    non_null_indices = data.dropna().index

    # Filter both data and images dataframes to keep only rows with non-null indices
    cleaned_data = data.loc[non_null_indices].reset_index(drop=True)
    cleaned_images = images.loc[non_null_indices].reset_index(drop=True)

    # Ensure the sizes of the cleaned dataframes match
    assert len(cleaned_data) == len(cleaned_images), "Mismatch in sizes of cleaned dataframes"

    return cleaned_data, cleaned_images