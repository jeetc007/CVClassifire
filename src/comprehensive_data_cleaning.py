import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ComprehensiveDataCleaner:
    def __init__(self):
        self.raw_data_path = Path("data/raw")
        self.processed_data_path = Path("data/processed")
        
    def load_raw_data(self):
        """Load raw dataset"""
        csv_files = list(self.raw_data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV file found in data/raw folder")
        
        dataset_path = csv_files[0]
        print(f"Loading raw data from: {dataset_path}")
        return pd.read_csv(dataset_path)
    
    def analyze_data_quality(self, df):
        """Comprehensive data quality analysis"""
        print("\n=== DATA QUALITY ANALYSIS ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print("\nMissing values:")
        missing_data = df.isnull().sum()
        for col, missing in missing_data.items():
            if missing > 0:
                print(f"  {col}: {missing:,} ({missing/len(df)*100:.1f}%)")
        
        print("\nData types:")
        print(df.dtypes)
        
        print("\nBasic statistics:")
        print(df.describe())
        
        # Check for negative values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print("\nNegative values check:")
        for col in numeric_cols:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"  {col}: {negative_count:,} negative values")
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicate_count:,}")
        
        return missing_data, duplicate_count
    
    def clean_data_comprehensive(self, df):
        """Comprehensive data cleaning"""
        print("\n=== COMPREHENSIVE DATA CLEANING ===")
        
        # Make a copy for cleaning
        df_clean = df.copy()
        initial_rows = len(df_clean)
        print(f"Starting with {initial_rows:,} rows")
        
        # 1. Handle missing Customer ID (critical for customer analysis)
        missing_customer_before = df_clean['Customer ID'].isnull().sum()
        df_clean = df_clean.dropna(subset=['Customer ID'])
        removed_missing_customer = missing_customer_before - df_clean['Customer ID'].isnull().sum()
        print(f"1. Removed {removed_missing_customer:,} rows with missing Customer ID")
        
        # 2. Handle missing values in critical columns
        critical_cols = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price']
        for col in critical_cols:
            missing_before = df_clean[col].isnull().sum()
            if missing_before > 0:
                df_clean = df_clean.dropna(subset=[col])
                print(f"2. Removed {missing_before:,} rows with missing {col}")
        
        # 3. Remove invalid quantities (0, negative)
        invalid_quantity_before = len(df_clean[(df_clean['Quantity'] <= 0)])
        df_clean = df_clean[df_clean['Quantity'] > 0]
        print(f"3. Removed {invalid_quantity_before:,} rows with non-positive Quantity")
        
        # 4. Remove invalid prices (0, negative)
        invalid_price_before = len(df_clean[(df_clean['Price'] <= 0)])
        df_clean = df_clean[df_clean['Price'] > 0]
        print(f"4. Removed {invalid_price_before:,} rows with non-positive Price")
        
        # 5. Remove duplicate rows
        duplicate_before = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        print(f"5. Removed {duplicate_before:,} duplicate rows")
        
        # 6. Clean and standardize data types
        print("6. Standardizing data types...")
        
        # Convert Customer ID to integer (handle any float values)
        df_clean['Customer ID'] = pd.to_numeric(df_clean['Customer ID'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Customer ID'])
        df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)
        
        # Convert InvoiceDate to datetime
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')
        df_clean = df_clean.dropna(subset=['InvoiceDate'])
        
        # 7. Clean text fields
        print("7. Cleaning text fields...")
        
        # Remove leading/trailing whitespace from descriptions
        df_clean['Description'] = df_clean['Description'].str.strip()
        
        # Remove rows with empty descriptions after cleaning
        empty_desc = df_clean['Description'].str.len() == 0
        empty_desc_count = empty_desc.sum()
        if empty_desc_count > 0:
            df_clean = df_clean[~empty_desc]
            print(f"   Removed {empty_desc_count:,} rows with empty descriptions")
        
        # 8. Create derived features for validation
        print("8. Creating validation features...")
        df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['Price']
        
        # 9. Remove outliers (extreme values)
        print("9. Removing outliers...")
        
        # Remove transactions with extremely high amounts (likely errors)
        amount_threshold = df_clean['TotalAmount'].quantile(0.999)
        extreme_amounts = df_clean['TotalAmount'] > amount_threshold
        extreme_count = extreme_amounts.sum()
        if extreme_count > 0:
            print(f"   Removed {extreme_count:,} rows with extreme amounts (> {amount_threshold:.2f})")
            df_clean = df_clean[~extreme_amounts]
        
        # 10. Final validation
        final_rows = len(df_clean)
        total_removed = initial_rows - final_rows
        
        print(f"\n=== CLEANING SUMMARY ===")
        print(f"Initial rows: {initial_rows:,}")
        print(f"Final rows: {final_rows:,}")
        print(f"Total removed: {total_removed:,} ({total_removed/initial_rows*100:.1f}%)")
        print(f"Data retention: {final_rows/initial_rows*100:.1f}%")
        
        # Final data quality checks
        print(f"\n=== FINAL DATA QUALITY CHECKS ===")
        print(f"No missing Customer IDs: {df_clean['Customer ID'].isnull().sum() == 0}")
        print(f"All quantities positive: {(df_clean['Quantity'] > 0).all()}")
        print(f"All prices positive: {(df_clean['Price'] > 0).all()}")
        print(f"No duplicates: {not df_clean.duplicated().any()}")
        print(f"Valid dates: {df_clean['InvoiceDate'].notna().all()}")
        
        return df_clean
    
    def create_cleaned_customer_features(self, df_clean):
        """Create customer-level features from cleaned data"""
        print("\n=== CREATING CUSTOMER FEATURES ===")
        
        # Group by Customer ID
        customer_features = df_clean.groupby('Customer ID').agg({
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
        
        # Handle customers with only one transaction
        single_transaction = customer_features['TransactionCount'] == 1
        customer_features.loc[single_transaction, 'AvgDaysBetweenPurchases'] = 0
        
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
        
        print(f"K-Means clustering completed with 4 clusters")
        print(f"Cluster centers (TotalAmount_Sum): {cluster_amounts[cluster_order]}")
        
        # Encode Country (frequency encoding)
        country_counts = df_clean['Country'].value_counts()
        customer_features['Country_Encoded'] = customer_features['Country'].map(country_counts)
        
        # Handle datetime columns - convert to numeric features
        customer_features['FirstPurchase_Year'] = customer_features['FirstPurchase'].dt.year
        customer_features['FirstPurchase_Month'] = customer_features['FirstPurchase'].dt.month
        customer_features['LastPurchase_Year'] = customer_features['LastPurchase'].dt.year
        customer_features['LastPurchase_Month'] = customer_features['LastPurchase'].dt.month
        
        # Drop original datetime and categorical columns
        customer_features = customer_features.drop(['FirstPurchase', 'LastPurchase', 'Country', 'CustomerSegment'], axis=1)
        
        # Handle any remaining missing values (only for numeric columns)
        numeric_cols = customer_features.select_dtypes(include=[np.number]).columns
        customer_features[numeric_cols] = customer_features[numeric_cols].fillna(customer_features[numeric_cols].median())
        
        # Remove outliers from customer-level data
        print("10. Removing customer-level outliers...")
        
        # Remove outliers based on TotalAmount_Sum using very aggressive IQR method
        Q1 = customer_features['TotalAmount_Sum'].quantile(0.25)
        Q3 = customer_features['TotalAmount_Sum'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (customer_features['TotalAmount_Sum'] < lower_bound) | (customer_features['TotalAmount_Sum'] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            print(f"   Removed {outlier_count:,} customers with outlier TotalAmount_Sum")
            print(f"   Outlier range: < {lower_bound:.2f} or > {upper_bound:.2f}")
            customer_features = customer_features[~outlier_mask]
        
        # Additional aggressive outlier removal using percentile method
        # Remove top 5% of customers by TotalAmount_Sum
        percentile_95 = customer_features['TotalAmount_Sum'].quantile(0.95)
        extreme_high_mask = customer_features['TotalAmount_Sum'] > percentile_95
        extreme_high_count = extreme_high_mask.sum()
        
        if extreme_high_count > 0:
            print(f"   Removed {extreme_high_count:,} customers in top 5% by TotalAmount_Sum")
            print(f"   95th percentile threshold: {percentile_95:.2f}")
            customer_features = customer_features[~extreme_high_mask]
        
        # Even more aggressive - remove top 3% by TotalAmount_Sum
        percentile_97 = customer_features['TotalAmount_Sum'].quantile(0.97)
        extreme_high_mask_97 = customer_features['TotalAmount_Sum'] > percentile_97
        extreme_high_count_97 = extreme_high_mask_97.sum()
        
        if extreme_high_count_97 > 0:
            print(f"   Removed {extreme_high_count_97:,} customers in top 3% by TotalAmount_Sum")
            print(f"   97th percentile threshold: {percentile_97:.2f}")
            customer_features = customer_features[~extreme_high_mask_97]
        
        # Also remove outliers for TransactionCount with aggressive percentile removal
        # Remove top 5% by TransactionCount
        percentile_95_tc = customer_features['TransactionCount'].quantile(0.95)
        extreme_tc_mask = customer_features['TransactionCount'] > percentile_95_tc
        extreme_tc_count = extreme_tc_mask.sum()
        
        if extreme_tc_count > 0:
            print(f"   Removed {extreme_tc_count:,} customers in top 5% by TransactionCount")
            print(f"   95th percentile threshold: {percentile_95_tc:.0f}")
            customer_features = customer_features[~extreme_tc_mask]
        
        # Remove outliers for Quantity_Sum with aggressive percentile removal
        # Remove top 5% by Quantity_Sum
        percentile_95_qs = customer_features['Quantity_Sum'].quantile(0.95)
        extreme_qs_mask = customer_features['Quantity_Sum'] > percentile_95_qs
        extreme_qs_count = extreme_qs_mask.sum()
        
        if extreme_qs_count > 0:
            print(f"   Removed {extreme_qs_count:,} customers in top 5% by Quantity_Sum")
            print(f"   95th percentile threshold: {percentile_95_qs:.0f}")
            customer_features = customer_features[~extreme_qs_mask]
        
        # Remove outliers for UniqueProducts with aggressive percentile removal
        # Remove top 5% by UniqueProducts
        percentile_95_up = customer_features['UniqueProducts'].quantile(0.95)
        extreme_up_mask = customer_features['UniqueProducts'] > percentile_95_up
        extreme_up_count = extreme_up_mask.sum()
        
        if extreme_up_count > 0:
            print(f"   Removed {extreme_up_count:,} customers in top 5% by UniqueProducts")
            print(f"   95th percentile threshold: {percentile_95_up:.0f}")
            customer_features = customer_features[~extreme_up_mask]
        
        print(f"Created features for {len(customer_features)} customers")
        print(f"Customer segment distribution:")
        segment_dist = customer_features['CustomerSegment_Encoded'].value_counts().sort_index()
        segment_names = ['Low Value', 'Medium Value', 'High Value', 'VIP']
        for seg, count in segment_dist.items():
            print(f"  {segment_names[seg]}: {count:,} customers")
        
        return customer_features
    
    def save_cleaned_data(self, df_clean, customer_features):
        """Save cleaned datasets"""
        print("\n=== SAVING CLEANED DATA ===")
        
        # Create directories
        self.processed_data_path.mkdir(exist_ok=True)
        
        # Save cleaned transaction data
        cleaned_transactions_path = self.processed_data_path / "cleaned_transactions.csv"
        df_clean.to_csv(cleaned_transactions_path, index=False)
        print(f"Cleaned transactions saved: {cleaned_transactions_path}")
        
        # Save customer features
        customer_features_path = self.processed_data_path / "cleaned_customer_features.csv"
        customer_features.to_csv(customer_features_path, index=False)
        print(f"Customer features saved: {customer_features_path}")
        
        return cleaned_transactions_path, customer_features_path
    
    def generate_data_quality_report(self, df_original, df_clean, customer_features):
        """Generate comprehensive data quality report"""
        print("\n=== DATA QUALITY REPORT ===")
        
        report_lines = []
        report_lines.append("DATA CLEANING AND QUALITY REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Original dataset: {df_original.shape[0]:,} rows")
        report_lines.append(f"Cleaned dataset: {df_clean.shape[0]:,} rows")
        report_lines.append(f"Data retention: {(len(df_clean)/len(df_original)*100):.1f}%")
        report_lines.append(f"Unique customers: {customer_features.shape[0]:,}")
        
        report_lines.append("\nCustomer Segments:")
        segment_dist = customer_features['CustomerSegment_Encoded'].value_counts().sort_index()
        segment_names = ['Low Value', 'Medium Value', 'High Value', 'VIP']
        for seg, count in segment_dist.items():
            percentage = count / len(customer_features) * 100
            report_lines.append(f"  {segment_names[seg]}: {count:,} ({percentage:.1f}%)")
        
        report_lines.append("\nKey Metrics:")
        report_lines.append(f"  Average total amount: ${customer_features['TotalAmount_Sum'].mean():.2f}")
        report_lines.append(f"  Average transaction count: {customer_features['TransactionCount'].mean():.1f}")
        report_lines.append(f"  Average unique products: {customer_features['UniqueProducts'].mean():.1f}")
        report_lines.append(f"  Average customer tenure: {customer_features['CustomerTenureDays'].mean():.1f} days")
        
        # Save report
        report_path = self.processed_data_path / "data_quality_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Data quality report saved: {report_path}")
        
        # Print report
        for line in report_lines:
            print(line)
    
    def run_comprehensive_cleaning(self):
        """Run the complete data cleaning pipeline"""
        print("=" * 60)
        print("COMPREHENSIVE DATA CLEANING PIPELINE")
        print("=" * 60)
        
        # Load raw data
        raw_df = self.load_raw_data()
        
        # Analyze data quality
        missing_data, duplicate_count = self.analyze_data_quality(raw_df)
        
        # Clean data comprehensively
        cleaned_df = self.clean_data_comprehensive(raw_df)
        
        # Create customer features
        customer_features = self.create_cleaned_customer_features(cleaned_df)
        
        # Save cleaned data
        cleaned_path, features_path = self.save_cleaned_data(cleaned_df, customer_features)
        
        # Generate quality report
        self.generate_data_quality_report(raw_df, cleaned_df, customer_features)
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE DATA CLEANING COMPLETED!")
        print("=" * 60)
        
        return cleaned_df, customer_features

if __name__ == "__main__":
    cleaner = ComprehensiveDataCleaner()
    cleaned_data, customer_features = cleaner.run_comprehensive_cleaning()
