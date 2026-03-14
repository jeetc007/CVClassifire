import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class Predictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.artifacts_path = Path("artifacts/models")
        self.segment_mapping = {0: 'Low Value', 1: 'Medium Value', 2: 'High Value'}
        
    def load_artifacts(self):
        """Load all saved artifacts"""
        print("Loading prediction artifacts...")
        
        try:
            # Load model
            model_path = self.artifacts_path / "model.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                print(f"Model loaded from: {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
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
                print(f"Selected features loaded: {self.selected_features}")
            
            # Load model metadata
            metadata_path = self.artifacts_path / "model_metadata.pkl"
            if metadata_path.exists():
                metadata = joblib.load(metadata_path)
                print(f"Model metadata loaded: {metadata['model_name']}")
            
            print("All artifacts loaded successfully!")
            
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            raise
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        print("Preprocessing input data...")
        
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Ensure CustomerID is present (add dummy if missing)
        if 'CustomerID' not in input_df.columns:
            input_df['CustomerID'] = 0
        
        # Create derived features (same as training)
        input_df['AvgTransactionValue'] = input_df['TotalAmount_Sum'] / input_df['TransactionCount']
        input_df['ProductsPerTransaction'] = input_df['UniqueProducts'] / input_df['TransactionCount']
        input_df['QuantityPerTransaction'] = input_df['Quantity_Sum'] / input_df['TransactionCount']
        input_df['ValuePerProduct'] = input_df['TotalAmount_Sum'] / input_df['UniqueProducts']
        input_df['PurchaseFrequency'] = input_df['TransactionCount'] / (input_df['CustomerTenureDays'] + 1)
        input_df['PriceSensitivity'] = input_df['Price_Mean'] / (input_df['Quantity_Mean'] + 1)
        input_df['LoyaltyScore'] = (input_df['TransactionCount'] * np.log1p(input_df['CustomerTenureDays'])) / 100
        
        # Handle any infinite values
        input_df = input_df.replace([np.inf, -np.inf], np.nan)
        input_df = input_df.fillna(input_df.median())
        
        print(f"Input data preprocessed. Shape: {input_df.shape}")
        return input_df
    
    def select_features(self, input_df):
        """Select features using the saved feature selector"""
        print("Selecting features...")
        
        # Remove CustomerID (it's just an identifier)
        if 'CustomerID' in input_df.columns:
            features_df = input_df.drop('CustomerID', axis=1)
        else:
            features_df = input_df.copy()
        
        # The feature selector expects all 21 features in the correct order
        # These are the features that were present during training (excluding CustomerID)
        expected_features = [
            'TotalAmount_Sum', 'TotalAmount_Mean', 'TransactionCount', 'Quantity_Sum', 
            'Quantity_Mean', 'Price_Mean', 'UniqueProducts', 'CustomerTenureDays', 
            'AvgDaysBetweenPurchases', 'Country_Encoded', 'FirstPurchase_Year', 
            'FirstPurchase_Month', 'LastPurchase_Year', 'LastPurchase_Month', 
            'AvgTransactionValue', 'ProductsPerTransaction', 'QuantityPerTransaction', 
            'ValuePerProduct', 'PurchaseFrequency', 'PriceSensitivity', 'LoyaltyScore'
        ]
        
        # Ensure we have all expected features in the correct order
        features_for_selector = features_df[expected_features]
        
        if self.feature_selector is not None:
            # Use feature selector to transform data
            selected_features_transformed = self.feature_selector.transform(features_for_selector)
            selected_df = pd.DataFrame(selected_features_transformed, columns=self.selected_features)
        else:
            # If no feature selector, use the saved feature list
            selected_df = features_df[self.selected_features]
        return selected_df
    
    def scale_features(self, features_df):
        """Scale features using the saved scaler"""
        print("Scaling features...")
        
        if self.scaler is not None:
            scaled_features = self.scaler.transform(features_df)
            scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns)
        else:
            scaled_df = features_df.copy()
        
        print("Features scaled successfully")
        return scaled_df
    
    def predict(self, input_data):
        """Make prediction on input data"""
        print("Making prediction...")
        
        # Load artifacts if not already loaded
        if self.model is None:
            self.load_artifacts()
        
        # Preprocess input
        preprocessed_df = self.preprocess_input(input_data)
        
        # Select features
        selected_df = self.select_features(preprocessed_df)
        
        # Scale features
        scaled_df = self.scale_features(selected_df)
        
        # Make prediction
        prediction = self.model.predict(scaled_df)
        
        # Get prediction probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(scaled_df)
        else:
            probabilities = None
        
        # Convert prediction to segment name
        segment_names = [self.segment_mapping[pred] for pred in prediction]
        
        print("Prediction completed!")
        
        return {
            'prediction': prediction[0],
            'segment': segment_names[0],
            'probabilities': probabilities[0].tolist() if probabilities is not None else None,
            'segment_mapping': self.segment_mapping
        }
    
    def batch_predict(self, input_data_list):
        """Make predictions on multiple inputs"""
        print(f"Making batch predictions for {len(input_data_list)} inputs...")
        
        results = []
        for input_data in input_data_list:
            result = self.predict(input_data)
            results.append(result)
        
        print(f"Batch predictions completed for {len(results)} inputs")
        return results
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.selected_features, self.model.feature_importances_))
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return sorted_importance
        else:
            return None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            self.load_artifacts()
        
        info = {
            'model_type': type(self.model).__name__,
            'selected_features': self.selected_features,
            'num_features': len(self.selected_features) if self.selected_features else 0,
            'segment_mapping': self.segment_mapping
        }
        
        # Add model-specific info
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        if hasattr(self.model, 'max_depth'):
            info['max_depth'] = self.model.max_depth
        if hasattr(self.model, 'n_neighbors'):
            info['n_neighbors'] = self.model.n_neighbors
        
        return info

# Example usage and testing
if __name__ == "__main__":
    # Test the predictor
    predictor = Predictor()
    
    # Load artifacts
    predictor.load_artifacts()
    
    # Create sample input data (you would get this from user input)
    sample_input = {
        'TotalAmount_Sum': 1000.0,
        'TotalAmount_Mean': 50.0,
        'TransactionCount': 20,
        'Quantity_Sum': 100,
        'Quantity_Mean': 5,
        'Price_Mean': 10.0,
        'Country_Encoded': 100,
        'UniqueProducts': 15,
        'CustomerTenureDays': 365,
        'AvgDaysBetweenPurchases': 18,
        'FirstPurchase_Year': 2023,
        'FirstPurchase_Month': 1,
        'LastPurchase_Year': 2023,
        'LastPurchase_Month': 12
    }
    
    # Make prediction
    try:
        result = predictor.predict(sample_input)
        print("\nPrediction Result:")
        print(f"Predicted Segment: {result['segment']}")
        print(f"Prediction (numeric): {result['prediction']}")
        if result['probabilities']:
            print(f"Probabilities: {result['probabilities']}")
        
        # Get model info
        model_info = predictor.get_model_info()
        print(f"\nModel Info: {model_info}")
        
        # Get feature importance if available
        importance = predictor.get_feature_importance()
        if importance:
            print(f"\nTop 5 Feature Importances:")
            for i, (feature, imp) in enumerate(list(importance.items())[:5]):
                print(f"  {i+1}. {feature}: {imp:.4f}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Please make sure you have run the training pipeline first: python src/train_model.py")
