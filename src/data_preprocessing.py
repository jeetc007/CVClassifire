import pandas as pd
import numpy as np
from pathlib import Path
import os

class DataPreprocessor:
    def __init__(self):
        self.raw_data_path = Path("data/raw")
        self.processed_data_path = Path("data/processed")
        
    def load_dataset(self):
        """Load dataset from data/raw folder"""
        csv_files = list(self.raw_data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV file found in data/raw folder")
        
        # Load the first CSV file found
        dataset_path = csv_files[0]
        print(f"Loading dataset from: {dataset_path}")
        return pd.read_csv(dataset_path)
    
    def clean_data(self, df):
        """Clean the dataset"""
        print("Starting data cleaning...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove rows with missing Customer ID (essential for customer analysis)
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['Customer ID'])
        print(f"Removed {initial_rows - len(df_clean)} rows with missing Customer ID")
        
        # Remove rows with missing critical values
        df_clean = df_clean.dropna(subset=['Quantity', 'Price', 'InvoiceDate'])
        
        # Filter out negative quantities and prices (returns and errors)
        df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]
        
        # Remove duplicates
        duplicates_removed = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {duplicates_removed} duplicate rows")
        
        # Convert Customer ID to integer
        df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)
        
        # Convert InvoiceDate to datetime
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
        
        # Create total amount column
        df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['Price']
        
        print(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def create_customer_features(self, df):
        """Create customer-level features from transaction data"""
        print("Creating customer-level features...")
        
        # Group by Customer ID
        customer_features = df.groupby('Customer ID').agg({
            'TotalAmount': ['sum', 'mean', 'count'],
            'Quantity': ['sum', 'mean'],
            'Price': ['mean'],
            'InvoiceDate': ['min', 'max'],
            'Country': 'first',
            'StockCode': 'nunique'
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = [
            'CustomerID', 'TotalAmount_Sum', 'TotalAmount_Mean', 'TransactionCount',
            'Quantity_Sum', 'Quantity_Mean', 'Price_Mean',
            'FirstPurchase', 'LastPurchase', 'Country', 'UniqueProducts'
        ]
        
        # Calculate customer tenure in days
        customer_features['CustomerTenureDays'] = (
            customer_features['LastPurchase'] - customer_features['FirstPurchase']
        ).dt.days
        
        # Calculate average days between purchases
        customer_features['AvgDaysBetweenPurchases'] = customer_features['CustomerTenureDays'] / (
            customer_features['TransactionCount'] - 1
        ).replace([np.inf, -np.inf], np.nan)
        
        # Create customer segments using K-Means clustering (4 tiers: Low/Medium/High/VIP)
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Select features for clustering
        clustering_features = ['TotalAmount_Sum', 'TransactionCount', 'UniqueProducts', 'CustomerTenureDays']
        clustering_data = customer_features[clustering_features].copy()
        
        # Handle any remaining NaN values
        clustering_data = clustering_data.fillna(clustering_data.mean())
        
        # Standardize features for clustering
        scaler = StandardScaler()
        clustering_scaled = scaler.fit_transform(clustering_data)
        
        # Apply K-Means with 4 clusters
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(clustering_scaled)
        
        # Assign cluster labels based on cluster centroids (sorted by TotalAmount_Sum)
        cluster_centers = kmeans.cluster_centers_
        cluster_amounts = cluster_centers[:, 0]  # TotalAmount_Sum is first feature
        cluster_order = np.argsort(cluster_amounts)
        
        # Map cluster numbers to tier labels
        tier_mapping = {}
        tier_labels = ['Low Value', 'Medium Value', 'High Value', 'VIP']
        for i, cluster_num in enumerate(cluster_order):
            tier_mapping[cluster_num] = tier_labels[i]
        
        # Create CustomerSegment and CustomerSegment_Encoded
        customer_features['CustomerSegment'] = [tier_mapping[label] for label in cluster_labels]
        segment_mapping = {'Low Value': 0, 'Medium Value': 1, 'High Value': 2, 'VIP': 3}
        customer_features['CustomerSegment_Encoded'] = customer_features['CustomerSegment'].map(segment_mapping)
        
        print(f"Created features for {len(customer_features)} customers")
        return customer_features
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        # Encode Country (frequency encoding)
        country_counts = df_encoded['Country'].value_counts()
        df_encoded['Country_Encoded'] = df_encoded['Country'].map(country_counts)
        
        # Encode CustomerSegment (target variable)
        segment_mapping = {'Low Value': 0, 'Medium Value': 1, 'High Value': 2, 'VIP': 3}
        df_encoded['CustomerSegment_Encoded'] = df_encoded['CustomerSegment'].map(segment_mapping)
        
        # Drop original categorical columns
        df_encoded = df_encoded.drop(['Country', 'CustomerSegment'], axis=1)
        
        # Handle datetime columns - convert to numeric features
        df_encoded['FirstPurchase_Year'] = df_encoded['FirstPurchase'].dt.year
        df_encoded['FirstPurchase_Month'] = df_encoded['FirstPurchase'].dt.month
        df_encoded['LastPurchase_Year'] = df_encoded['LastPurchase'].dt.year
        df_encoded['LastPurchase_Month'] = df_encoded['LastPurchase'].dt.month
        
        # Drop original datetime columns
        df_encoded = df_encoded.drop(['FirstPurchase', 'LastPurchase'], axis=1)
        
        print("Categorical encoding completed")
        return df_encoded
    
    def save_processed_data(self, df, filename="processed_customer_data.csv"):
        """Save processed dataset"""
        os.makedirs(self.processed_data_path, exist_ok=True)
        file_path = self.processed_data_path / filename
        
        df.to_csv(file_path, index=False)
        print(f"Processed data saved to: {file_path}")
        return file_path
    
    def preprocess_pipeline(self):
        """Complete preprocessing pipeline"""
        print("Starting data preprocessing pipeline...")
        
        # Load and clean data
        raw_df = self.load_dataset()
        cleaned_df = self.clean_data(raw_df)
        
        # Create customer features
        customer_df = self.create_customer_features(cleaned_df)
        
        # Encode categorical features
        processed_df = self.encode_categorical_features(customer_df)
        
        # Save processed data
        saved_path = self.save_processed_data(processed_df)
        
        print("Preprocessing pipeline completed successfully!")
        return processed_df, saved_path

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    processed_data, path = preprocessor.preprocess_pipeline()
    print(f"\nFinal processed data shape: {processed_data.shape}")
    print(f"Columns: {list(processed_data.columns)}")
    print(f"Data saved to: {path}")
