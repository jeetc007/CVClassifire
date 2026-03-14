import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from pathlib import Path

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.artifacts_path = Path("artifacts/models")
        
    def create_derived_features(self, df):
        """Create additional derived features"""
        print("Creating derived features...")
        
        df_fe = df.copy()
        
        # Ratio features
        df_fe['AvgTransactionValue'] = df_fe['TotalAmount_Sum'] / df_fe['TransactionCount']
        df_fe['ProductsPerTransaction'] = df_fe['UniqueProducts'] / df_fe['TransactionCount']
        df_fe['QuantityPerTransaction'] = df_fe['Quantity_Sum'] / df_fe['TransactionCount']
        
        # Value per product
        df_fe['ValuePerProduct'] = df_fe['TotalAmount_Sum'] / df_fe['UniqueProducts']
        
        # Purchase frequency indicator
        df_fe['PurchaseFrequency'] = df_fe['TransactionCount'] / (df_fe['CustomerTenureDays'] + 1)
        
        # Price sensitivity (average price vs quantity ratio)
        df_fe['PriceSensitivity'] = df_fe['Price_Mean'] / (df_fe['Quantity_Mean'] + 1)
        
        # Customer loyalty score (based on transaction count and tenure)
        df_fe['LoyaltyScore'] = (df_fe['TransactionCount'] * np.log1p(df_fe['CustomerTenureDays'])) / 100
        
        # Handle any infinite values
        df_fe = df_fe.replace([np.inf, -np.inf], np.nan)
        df_fe = df_fe.fillna(df_fe.median())
        
        print(f"Created {df_fe.shape[1] - df.shape[1]} new derived features")
        return df_fe
    
    def select_features(self, X, y, k=15):
        """Select top k features using statistical tests"""
        print(f"Selecting top {k} features...")
        
        # Remove CustomerID from features (it's just an identifier)
        if 'CustomerID' in X.columns:
            X = X.drop('CustomerID', axis=1)
        
        # Use SelectKBest with f_classif
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = X.columns[selected_indices].tolist()
        
        print(f"Selected features: {self.selected_features}")
        
        # Create DataFrame with selected features
        X_selected_df = pd.DataFrame(X_selected, columns=self.selected_features)
        
        return X_selected_df
    
    def scale_features(self, X, fit=True):
        """Scale features using StandardScaler"""
        print("Scaling features...")
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            print("Scaler fitted and transformed data")
        else:
            X_scaled = self.scaler.transform(X)
            print("Data transformed using existing scaler")
        
        # Convert back to DataFrame
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled_df
    
    def save_artifacts(self):
        """Save scaler and feature selector"""
        print("Saving feature engineering artifacts...")
        
        # Create artifacts directory if it doesn't exist
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        scaler_path = self.artifacts_path / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save feature selector if it exists
        if self.feature_selector is not None:
            selector_path = self.artifacts_path / "feature_selector.pkl"
            joblib.dump(self.feature_selector, selector_path)
            print(f"Feature selector saved to: {selector_path}")
        
        # Save selected features list
        if self.selected_features is not None:
            features_path = self.artifacts_path / "selected_features.pkl"
            joblib.dump(self.selected_features, features_path)
            print(f"Selected features saved to: {features_path}")
    
    def load_artifacts(self):
        """Load scaler and feature selector"""
        print("Loading feature engineering artifacts...")
        
        # Load scaler
        scaler_path = self.artifacts_path / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from: {scaler_path}")
        
        # Load feature selector
        selector_path = self.artifacts_path / "feature_selector.pkl"
        if selector_path.exists():
            self.feature_selector = joblib.load(selector_path)
            print(f"Feature selector loaded from: {selector_path}")
        
        # Load selected features
        features_path = self.artifacts_path / "selected_features.pkl"
        if features_path.exists():
            self.selected_features = joblib.load(features_path)
            print(f"Selected features loaded from: {features_path}")
    
    def feature_engineering_pipeline(self, df, target_column='CustomerSegment_Encoded', fit_scaler=True):
        """Complete feature engineering pipeline"""
        print("Starting feature engineering pipeline...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Create derived features
        X_fe = self.create_derived_features(X)
        
        # Remove target-leaking features that were used to create the segments in data_preprocessing.py
        target_leak_columns = ['TotalAmount_Sum']
        X_fe = X_fe.drop(columns=[col for col in target_leak_columns if col in X_fe.columns])
        
        # Select features
        X_selected = self.select_features(X_fe, y)
        
        # Scale features
        X_scaled = self.scale_features(X_selected, fit=fit_scaler)
        
        # Save artifacts if fitting
        if fit_scaler:
            self.save_artifacts()
        
        print("Feature engineering pipeline completed!")
        print(f"Final feature set shape: {X_scaled.shape}")
        
        return X_scaled, y

if __name__ == "__main__":
    # Test the feature engineering pipeline
    from data_preprocessing import DataPreprocessor
    
    # Load preprocessed data
    preprocessor = DataPreprocessor()
    processed_df, _ = preprocessor.preprocess_pipeline()
    
    # Apply feature engineering
    fe = FeatureEngineer()
    X_final, y_final = fe.feature_engineering_pipeline(processed_df)
    
    print(f"\nFinal processed features shape: {X_final.shape}")
    print(f"Target variable shape: {y_final.shape}")
    print(f"Selected features: {fe.selected_features}")
