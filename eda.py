import pandas as pd
import seaborn as sns
from load_data import load_data
from clean_data import clean_data
import os

import matplotlib.pyplot as plt

# Load the data
data, images = load_data()

print("----------------------")

# Display basic information about the dataset
print("Dataframe Info:")
print(data.info())

print("----------------------")

print("\nFirst 5 Rows of Data:")
print(data.head())

print("----------------------")


# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

print("----------------------")
# Apply the clean_data function
data, images = clean_data(data, images)

# Basic statistics of the dataset
print("\nDescriptive Statistics:")
print(data.describe(include='all'))  # Include all columns, even non-numerical ones

print("----------------------")

# Correlation matrix for numerical features
print("\nCorrelation Matrix:")
correlation_matrix = data.corr()
print(correlation_matrix)

print("----------------------")
# Perform EDA for gender, masterCategory, subCategory, articleType, baseColor, season, and usage
columns_to_analyze = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']

# Create a folder to save the visualizations
output_folder = "eda_visuals"
os.makedirs(output_folder, exist_ok=True)

# Create subfolders for each type of visualization
categories = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
for category in categories:
    os.makedirs(f"{output_folder}/{category}", exist_ok=True)

for column in columns_to_analyze:
    print(f"\nEDA for {column}:")
    print(data[column].value_counts())
    print("----------------------")
    
    # Plot the distribution of the column
    plt.figure(figsize=(8, 4))
    sns.countplot(data=data, x=column, order=data[column].value_counts().index)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig(f"{output_folder}/{column}/{column}_distribution.png")
    plt.close()

    # Compare the column with 'gender' if it's not 'gender'
    if column != 'gender':
        plt.figure(figsize=(8, 4))
        sns.countplot(data=data, x=column, hue='gender', order=data[column].value_counts().index)
        plt.title(f'{column} Distribution by Gender')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Gender')
        plt.savefig(f"{output_folder}/{column}/{column}_by_gender.png")
        plt.close()

# Plot the number of items in each subCategory, articleType, baseColor, and season under each masterCategory
master_categories = data['masterCategory'].unique()
for master_category in master_categories:
    for column in ['subCategory', 'articleType', 'baseColour', 'season']:
        plt.figure(figsize=(8, 4))
        sub_data = data[data['masterCategory'] == master_category]
        sns.countplot(data=sub_data, x=column, order=sub_data[column].value_counts().index)
        plt.title(f'Number of Items in Each {column} under {master_category}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.savefig(f"{output_folder}/{column}/{master_category}_{column}_distribution.png")
        plt.close()

# Plot the counts of each usage type
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x='usage', order=data['usage'].value_counts().index)
plt.title('Counts of Each Usage Type')
plt.xlabel('Usage')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig(f"{output_folder}/usage/usage_distribution.png")
plt.close()
