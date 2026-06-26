import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set design styles for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def plot_price_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plots the price distribution (both raw and log-scaled) to show positive skewness.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Raw price distribution (filtered to < $1000 for visualization clarity)
    sns.histplot(df[df['price'] < 1000]['price'], bins=50, kde=True, ax=axes[0], color='#2B6CB0')
    axes[0].set_title('Distribution of Prices (Under $1000/night)', fontweight='bold', pad=15)
    axes[0].set_xlabel('Price ($)')
    axes[0].set_ylabel('Count')
    
    # 2. Log-price distribution
    sns.histplot(df['log_price'], bins=50, kde=True, ax=axes[1], color='#319795')
    axes[1].set_title('Distribution of Log-Transformed Prices', fontweight='bold', pad=15)
    axes[1].set_xlabel('Log(Price + 1)')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'price_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved price distribution plot to {output_path}")

def plot_price_by_neighbourhood_group(df: pd.DataFrame, output_dir: str) -> None:
    """
    Creates box and violin plots of price by Borough (Neighbourhood Group).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter price to < $800 to avoid outlier distortion in boxplot
    df_filtered = df[df['price'] < 800]
    
    plt.figure(figsize=(12, 7))
    sns.violinplot(x='neighbourhood_group', y='price', data=df_filtered, 
                   palette='Set2', hue='neighbourhood_group', legend=False)
    
    plt.title('Airbnb Price Distribution by Borough (Price < $800)', fontweight='bold', pad=15)
    plt.xlabel('Borough (Neighbourhood Group)')
    plt.ylabel('Price per Night ($)')
    
    output_path = os.path.join(output_dir, 'price_by_borough.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved price by borough plot to {output_path}")

def plot_price_by_room_type(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plots the distribution of prices across different Room Types.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df_filtered = df[df['price'] < 800]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='room_type', y='price', data=df_filtered, palette='husl', hue='room_type', legend=False)
    
    plt.title('Airbnb Price Range by Room Type (Price < $800)', fontweight='bold', pad=15)
    plt.xlabel('Room Type')
    plt.ylabel('Price per Night ($)')
    
    output_path = os.path.join(output_dir, 'price_by_room_type.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved price by room type plot to {output_path}")

def plot_spatial_price_map(df: pd.DataFrame, output_dir: str) -> None:
    """
    Creates a scatter plot based on latitude/longitude, colored by log_price.
    This effectively reconstructs the map of New York City, showing price density.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    # We color by log_price to get a smooth gradients and avoid extreme outlier distortion
    scatter = plt.scatter(df['longitude'], df['latitude'], c=df['log_price'], 
                          cmap='inferno', s=4, alpha=0.5)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Log(Price + 1)', rotation=270, labelpad=15)
    
    # Annotate some borough centers for reference
    boroughs = {
        'Manhattan': (-73.9689, 40.7831),
        'Brooklyn': (-73.9442, 40.6782),
        'Queens': (-73.7949, 40.7282),
        'Bronx': (-73.8648, 40.8448),
        'Staten Island': (-74.1502, 40.5795)
    }
    for borough, (lon, lat) in boroughs.items():
        plt.text(lon, lat, borough, fontsize=12, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
        
    plt.title('Geographical Distribution of Airbnb Prices in New York City', fontweight='bold', pad=15)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    output_path = os.path.join(output_dir, 'spatial_price_map.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial price map to {output_path}")

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plots correlation heatmap for numerical features.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    numerical_cols = [
        'price', 'log_price', 'minimum_nights', 'number_of_reviews', 
        'reviews_per_month', 'calculated_host_listings_count', 
        'availability_365', 'name_length', 'days_since_last_review'
    ]
    
    # Filter numerical columns that actually exist in the dataframe
    cols_to_corr = [col for col in numerical_cols if col in df.columns]
    
    corr = df[cols_to_corr].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
    
    plt.title('Correlation Matrix of Numerical Features', fontweight='bold', pad=15)
    
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to {output_path}")
