"""
Comprehensive Customer Classification Prediction Pipeline

This module provides a complete pipeline for customer segmentation prediction
including data preprocessing, feature engineering, model loading, and prediction.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import logging
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomerPredictionPipeline:
    """
    Complete pipeline for customer segmentation prediction.
    
    This class handles:
    - Loading trained models and preprocessing artifacts
    - Data preprocessing and feature engineering
    - Feature selection and scaling
    - Making predictions with confidence scores
    - Batch processing capabilities
    """
    
    def __init__(self, model_dir: str = "artifacts/models"):
        """
        Initialize the prediction pipeline.
        
        Args:
            model_dir (str): Directory containing trained models and artifacts
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.model_metadata = None
        self.is_loaded = False
        
        # Feature engineering parameters
        self.expected_features = [
            'TotalAmount_Mean', 'TransactionCount', 'Quantity_Sum', 
            'Quantity_Mean', 'Price_Mean', 'UniqueProducts', 'CustomerTenureDays', 
            'AvgDaysBetweenPurchases', 'Country_Encoded', 'FirstPurchase_Year', 
            'FirstPurchase_Month', 'LastPurchase_Year', 'LastPurchase_Month', 
            'AvgTransactionValue', 'ProductsPerTransaction', 'QuantityPerTransaction', 
            'ValuePerProduct', 'PurchaseFrequency', 'PriceSensitivity', 'LoyaltyScore'
        ]
        
        # Segment mapping
        self.segment_mapping = {0: 'Low Value', 1: 'Medium Value', 2: 'High Value'}
        
        logger.info("CustomerPredictionPipeline initialized")
    
    def load_artifacts(self) -> bool:
        """
        Load all trained models and preprocessing artifacts.
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            logger.info("Loading prediction artifacts...")
            
            # Load model
            model_path = self.model_dir / "model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from: {model_path}")
            
            # Load scaler
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from: {scaler_path}")
            
            # Load feature selector
            selector_path = self.model_dir / "feature_selector.pkl"
            if selector_path.exists():
                self.feature_selector = joblib.load(selector_path)
                logger.info(f"Feature selector loaded from: {selector_path}")
            
            # Load selected features
            features_path = self.model_dir / "selected_features.pkl"
            if features_path.exists():
                self.selected_features = joblib.load(features_path)
                logger.info(f"Selected features loaded: {self.selected_features}")
            
            # Load model metadata
            metadata_path = self.model_dir / "model_metadata.pkl"
            if metadata_path.exists():
                self.model_metadata = joblib.load(metadata_path)
                logger.info(f"Model metadata loaded: {self.model_metadata.get('model_type', 'Unknown')}")
            
            self.is_loaded = True
            logger.info("All artifacts loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            return False
    
    def validate_input_data(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Validate and prepare input data for prediction.
        
        Args:
            data: Input data as dictionary or DataFrame
            
        Returns:
            pd.DataFrame: Validated input data
            
        Raises:
            ValueError: If required features are missing
        """
        if isinstance(data, dict):
            # Convert single customer dict to DataFrame
            df = pd.DataFrame([data])
        elif hasattr(data, 'columns') and hasattr(data, 'iloc'):  # Check if it's DataFrame-like
            df = data.copy()
        else:
            raise ValueError("Input data must be a dictionary or pandas DataFrame")
        
        # Check for required base features
        required_base_features = [
            'TotalAmount_Mean', 'TransactionCount', 'Quantity_Sum',
            'Quantity_Mean', 'Price_Mean', 'Country_Encoded', 'UniqueProducts',
            'CustomerTenureDays', 'FirstPurchase_Year', 'FirstPurchase_Month',
            'LastPurchase_Year', 'LastPurchase_Month'
        ]
        
        missing_features = [feat for feat in required_base_features if feat not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Validate data types and ranges
        numeric_features = df.select_dtypes(include=[np.number]).columns
        for col in numeric_features:
            if df[col].isnull().any():
                logger.warning(f"Missing values found in {col}, filling with median")
                df[col] = df[col].fillna(df[col].median())
            
            # Check for negative values where not expected
            if col in ['TransactionCount', 'Quantity_Sum', 'UniqueProducts']:
                if (df[col] <= 0).any():
                    logger.warning(f"Non-positive values found in {col}")
        
        return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for prediction.
        
        Args:
            df: Input DataFrame with base features
            
        Returns:
            pd.DataFrame: DataFrame with derived features added
        """
        logger.info("Creating derived features...")
        
        # Create derived features
        df['AvgDaysBetweenPurchases'] = np.where(
            df['TransactionCount'] > 1,
            df['CustomerTenureDays'] / (df['TransactionCount'] - 1),
            0
        )
        
        # Calculate average transaction value
        df['AvgTransactionValue'] = df['TotalAmount_Mean']
        
        # Calculate products per transaction
        df['ProductsPerTransaction'] = df['Quantity_Sum'] / df['TransactionCount']
        
        # Calculate quantity per transaction
        df['QuantityPerTransaction'] = df['Quantity_Mean']
        
        # Calculate value per product
        df['ValuePerProduct'] = np.where(
            df['Quantity_Sum'] > 0,
            (df['TotalAmount_Mean'] * df['TransactionCount']) / df['Quantity_Sum'],
            0
        )
        
        # Calculate purchase frequency (inverse of average days between purchases)
        df['PurchaseFrequency'] = np.where(
            df['AvgDaysBetweenPurchases'] > 0,
            365 / df['AvgDaysBetweenPurchases'],
            0
        )
        
        # Calculate price sensitivity (ratio of price mean to average transaction value)
        df['PriceSensitivity'] = np.where(
            df['AvgTransactionValue'] > 0,
            df['Price_Mean'] / df['AvgTransactionValue'],
            0
        )
        
        # Calculate loyalty score (combination of tenure and frequency)
        df['LoyaltyScore'] = (
            (df['CustomerTenureDays'] / 365) * 0.5 +  # Tenure component
            (df['PurchaseFrequency'] / 52) * 0.5      # Frequency component (weeks per year)
        )
        
        # Handle any infinite or NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        logger.info(f"Derived features created. Shape: {df.shape}")
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and order features for model prediction.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            pd.DataFrame: DataFrame with selected features in correct order
        """
        logger.info("Selecting features...")
        
        # Ensure we have all expected features
        missing_features = [feat for feat in self.expected_features if feat not in df.columns]
        if missing_features:
            raise ValueError(f"Missing expected features: {missing_features}")
        
        # Order features as expected by the feature selector
        features_for_selector = df[self.expected_features]
        
        if self.feature_selector is not None:
            # Apply feature selection
            selected_features_transformed = self.feature_selector.transform(features_for_selector)
            selected_df = pd.DataFrame(selected_features_transformed, columns=self.selected_features)
        else:
            # Use predefined feature list
            selected_df = df[self.selected_features]
        
        logger.info(f"Features selected. Shape: {selected_df.shape}")
        return selected_df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale features using the trained scaler.
        
        Args:
            df: DataFrame with selected features
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        logger.info("Scaling features...")
        
        if self.scaler is not None:
            scaled_features = self.scaler.transform(df)
            scaled_df = pd.DataFrame(scaled_features, columns=df.columns)
        else:
            # If no scaler, return as-is
            scaled_df = df.copy()
        
        logger.info("Features scaled successfully")
        return scaled_df
    
    def predict(self, data: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Make prediction for customer segmentation.
        
        Args:
            data: Input data as dictionary or DataFrame
            
        Returns:
            Dict: Prediction results with segment, probabilities, and metadata
        """
        if not self.is_loaded:
            if not self.load_artifacts():
                raise RuntimeError("Failed to load model artifacts")
        
        logger.info("Making prediction...")
        
        try:
            # Step 1: Validate input data
            input_df = self.validate_input_data(data)
            
            # Step 2: Create derived features
            df_with_derived = self.create_derived_features(input_df)
            
            # Step 3: Select features
            selected_features_df = self.select_features(df_with_derived)
            
            # Step 4: Scale features
            scaled_features_df = self.scale_features(selected_features_df)
            
            # Step 5: Make prediction
            prediction = self.model.predict(scaled_features_df)
            probabilities = self.model.predict_proba(scaled_features_df)
            
            # Convert prediction to segment name
            if len(prediction) == 1:
                # Single prediction
                segment = self.segment_mapping[int(prediction[0])]
                prob_dict = dict(zip(self.model.classes_, probabilities[0]))
                
                result = {
                    'prediction': int(prediction[0]),
                    'segment': segment,
                    'probabilities': probabilities[0].tolist(),
                    'probability_dict': prob_dict,
                    'confidence': float(np.max(probabilities[0])),
                    'timestamp': datetime.now().isoformat(),
                    'model_info': {
                        'model_type': self.model_metadata.get('model_type', 'Unknown') if self.model_metadata else 'Unknown',
                        'features_used': len(self.selected_features) if self.selected_features else 0
                    }
                }
            else:
                # Batch prediction
                segments = [self.segment_mapping[int(pred)] for pred in prediction]
                confidences = np.max(probabilities, axis=1)
                
                result = {
                    'predictions': prediction.tolist(),
                    'segments': segments,
                    'probabilities': probabilities.tolist(),
                    'confidences': confidences.tolist(),
                    'batch_size': len(prediction),
                    'timestamp': datetime.now().isoformat(),
                    'model_info': {
                        'model_type': self.model_metadata.get('model_type', 'Unknown') if self.model_metadata else 'Unknown',
                        'features_used': len(self.selected_features) if self.selected_features else 0
                    }
                }
            
            logger.info("Prediction completed successfully!")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_batch(self, data_list: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple customers.
        
        Args:
            data_list: List of customer data dictionaries
            
        Returns:
            List[Dict]: List of prediction results
        """
        logger.info(f"Making batch prediction for {len(data_list)} customers")
        
        # Convert to DataFrame
        df = pd.DataFrame(data_list)
        
        # Make batch prediction
        batch_result = self.predict(df)
        
        # Convert batch result to list of individual results
        individual_results = []
        for i in range(len(batch_result['predictions'])):
            result = {
                'prediction': batch_result['predictions'][i],
                'segment': batch_result['segments'][i],
                'probabilities': batch_result['probabilities'][i],
                'confidence': batch_result['confidences'][i],
                'timestamp': batch_result['timestamp'],
                'model_info': batch_result['model_info']
            }
            individual_results.append(result)
        
        logger.info(f"Batch prediction completed for {len(individual_results)} customers")
        return individual_results
    
    def get_feature_importance(self) -> Optional[Dict]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dict: Feature importance mapping or None if not available
        """
        if not self.is_loaded:
            if not self.load_artifacts():
                return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.selected_features, self.model.feature_importances_))
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return sorted_importance
        else:
            logger.warning("Model does not support feature importance")
            return None
    
    def get_pipeline_info(self) -> Dict:
        """
        Get information about the prediction pipeline.
        
        Returns:
            Dict: Pipeline information
        """
        return {
            'is_loaded': self.is_loaded,
            'model_type': self.model_metadata.get('model_type', 'Unknown') if self.model_metadata else 'Unknown',
            'total_features': len(self.expected_features),
            'selected_features': len(self.selected_features) if self.selected_features else 0,
            'feature_selector_available': self.feature_selector is not None,
            'scaler_available': self.scaler is not None,
            'segment_mapping': self.segment_mapping,
            'model_dir': str(self.model_dir)
        }


def create_sample_customer_data() -> Dict:
    """
    Create sample customer data for testing the pipeline.
    
    Returns:
        Dict: Sample customer data
    """
    return {
        'TotalAmount_Sum': 1500.0,
        'TotalAmount_Mean': 75.0,
        'TransactionCount': 20,
        'Quantity_Sum': 150,
        'Quantity_Mean': 7.5,
        'Price_Mean': 10.0,
        'Country_Encoded': 100,
        'UniqueProducts': 18,
        'CustomerTenureDays': 400,
        'FirstPurchase_Year': 2023,
        'FirstPurchase_Month': 1,
        'LastPurchase_Year': 2023,
        'LastPurchase_Month': 12
    }


def main():
    """
    Example usage of the prediction pipeline.
    """
    print("=== Customer Classification Prediction Pipeline Demo ===")
    
    # Initialize pipeline
    pipeline = CustomerPredictionPipeline()
    
    # Load artifacts
    if not pipeline.load_artifacts():
        print("Failed to load pipeline artifacts")
        return
    
    # Get pipeline info
    print("\nPipeline Info:")
    info = pipeline.get_pipeline_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test single prediction
    print("\n=== Single Prediction Test ===")
    sample_data = create_sample_customer_data()
    print(f"Input data: {sample_data}")
    
    try:
        result = pipeline.predict(sample_data)
        print(f"\nPrediction Result:")
        print(f"  Segment: {result['segment']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Probabilities: {result['probability_dict']}")
        print(f"  Model Type: {result['model_info']['model_type']}")
    except Exception as e:
        print(f"Prediction failed: {e}")
    
    # Test batch prediction
    print("\n=== Batch Prediction Test ===")
    batch_data = [
        create_sample_customer_data(),
        {
            'TotalAmount_Sum': 5000.0,
            'TotalAmount_Mean': 250.0,
            'TransactionCount': 20,
            'Quantity_Sum': 500,
            'Quantity_Mean': 25.0,
            'Price_Mean': 10.0,
            'Country_Encoded': 100,
            'UniqueProducts': 50,
            'CustomerTenureDays': 300,
            'FirstPurchase_Year': 2022,
            'FirstPurchase_Month': 1,
            'LastPurchase_Year': 2023,
            'LastPurchase_Month': 12
        },
        {
            'TotalAmount_Sum': 200.0,
            'TotalAmount_Mean': 10.0,
            'TransactionCount': 20,
            'Quantity_Sum': 40,
            'Quantity_Mean': 2.0,
            'Price_Mean': 5.0,
            'Country_Encoded': 100,
            'UniqueProducts': 8,
            'CustomerTenureDays': 100,
            'FirstPurchase_Year': 2023,
            'FirstPurchase_Month': 6,
            'LastPurchase_Year': 2023,
            'LastPurchase_Month': 12
        }
    ]
    
    try:
        batch_results = pipeline.predict_batch(batch_data)
        print(f"\nBatch Prediction Results ({len(batch_results)} customers):")
        for i, result in enumerate(batch_results):
            print(f"  Customer {i+1}: {result['segment']} (confidence: {result['confidence']:.3f})")
    except Exception as e:
        print(f"Batch prediction failed: {e}")
    
    # Get feature importance
    print("\n=== Feature Importance ===")
    importance = pipeline.get_feature_importance()
    if importance:
        print("Top 10 Important Features:")
        for i, (feature, score) in enumerate(list(importance.items())[:10]):
            print(f"  {i+1}. {feature}: {score:.4f}")
    else:
        print("Feature importance not available")
    
    print("\n=== Pipeline Demo Completed ===")


if __name__ == "__main__":
    main()
