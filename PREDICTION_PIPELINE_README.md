# Customer Classification Prediction Pipeline

A comprehensive, production-ready pipeline for customer segmentation prediction using machine learning.

## 🚀 Features

- **Robust Prediction Pipeline**: Complete ML pipeline with preprocessing, feature engineering, and prediction
- **Multiple Interfaces**: CLI, Python API, and REST API support
- **Batch Processing**: Handle single or multiple customer predictions efficiently
- **Data Validation**: Comprehensive input validation and error handling
- **Feature Engineering**: Automatic derived feature creation
- **Model Management**: Easy model loading and artifact management
- **Logging & Monitoring**: Detailed logging and pipeline health checks

## 📁 Files Overview

```
src/
├── prediction_pipeline.py    # Core prediction pipeline class
├── prediction_api.py         # Flask REST API wrapper
├── cli_predict.py           # Command-line interface
└── predict.py               # Original predictor (legacy)
```

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Ensure you're in the project directory
cd CVClassifire

# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### Model Training (Required First Step)
```bash
# Train the model if not already done
python src/train_model.py
```

## 🎯 Usage Examples

### 1. Python API Usage

```python
from src.prediction_pipeline import CustomerPredictionPipeline

# Initialize pipeline
pipeline = CustomerPredictionPipeline()

# Load model artifacts
pipeline.load_artifacts()

# Single prediction
customer_data = {
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

result = pipeline.predict(customer_data)
print(f"Customer Segment: {result['segment']}")
print(f"Confidence: {result['confidence']:.3f}")

# Batch prediction
batch_data = [customer_data, customer_data2, customer_data3]
results = pipeline.predict_batch(batch_data)
for i, result in enumerate(results):
    print(f"Customer {i+1}: {result['segment']}")
```

### 2. Command Line Interface

#### Single Prediction
```bash
# Using sample data
python src/cli_predict.py --single --input sample --verbose

# Using JSON file
python src/cli_predict.py --single --input customer_data.json

# Save results to file
python src/cli_predict.py --single --input sample --output result.json
```

#### Batch Prediction
```bash
# Using CSV file
python src/cli_predict.py --batch --input customers.csv --verbose

# Using JSON file
python src/cli_predict.py --batch --input customers.json --output results.json
```

#### Help & Options
```bash
python src/cli_predict.py --help
```

### 3. REST API

#### Start the API Server
```bash
python src/prediction_api.py
```

#### API Endpoints

**Health Check**
```bash
curl http://localhost:5000/health
```

**Single Prediction**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TotalAmount_Sum": 1500.0,
    "TotalAmount_Mean": 75.0,
    "TransactionCount": 20,
    "Quantity_Sum": 150,
    "Quantity_Mean": 7.5,
    "Price_Mean": 10.0,
    "Country_Encoded": 100,
    "UniqueProducts": 18,
    "CustomerTenureDays": 400,
    "FirstPurchase_Year": 2023,
    "FirstPurchase_Month": 1,
    "LastPurchase_Year": 2023,
    "LastPurchase_Month": 12
  }'
```

**Batch Prediction**
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '[{
    "TotalAmount_Sum": 1500.0,
    "TotalAmount_Mean": 75.0,
    "TransactionCount": 20,
    "Quantity_Sum": 150,
    "Quantity_Mean": 7.5,
    "Price_Mean": 10.0,
    "Country_Encoded": 100,
    "UniqueProducts": 18,
    "CustomerTenureDays": 400,
    "FirstPurchase_Year": 2023,
    "FirstPurchase_Month": 1,
    "LastPurchase_Year": 2023,
    "LastPurchase_Month": 12
  }, {
    "TotalAmount_Sum": 5000.0,
    "TotalAmount_Mean": 250.0,
    "TransactionCount": 20,
    "Quantity_Sum": 500,
    "Quantity_Mean": 25.0,
    "Price_Mean": 10.0,
    "Country_Encoded": 100,
    "UniqueProducts": 50,
    "CustomerTenureDays": 300,
    "FirstPurchase_Year": 2022,
    "FirstPurchase_Month": 1,
    "LastPurchase_Year": 2023,
    "LastPurchase_Month": 12
  }]'
```

**Pipeline Information**
```bash
curl http://localhost:5000/pipeline_info
curl http://localhost:5000/feature_importance
curl http://localhost:5000/sample_data
```

## 📊 Input Data Format

### Required Fields
All predictions require the following base fields:

```python
{
    "TotalAmount_Sum": float,        # Total amount spent by customer
    "TotalAmount_Mean": float,       # Average transaction amount
    "TransactionCount": int,         # Number of transactions
    "Quantity_Sum": int,            # Total quantity purchased
    "Quantity_Mean": float,         # Average quantity per transaction
    "Price_Mean": float,            # Average price per item
    "Country_Encoded": int,         # Encoded country value
    "UniqueProducts": int,          # Number of unique products
    "CustomerTenureDays": int,      # Customer tenure in days
    "FirstPurchase_Year": int,      # Year of first purchase
    "FirstPurchase_Month": int,     # Month of first purchase
    "LastPurchase_Year": int,       # Year of last purchase
    "LastPurchase_Month": int       # Month of last purchase
}
```

### Derived Features (Automatically Calculated)
The pipeline automatically creates these derived features:
- `AvgDaysBetweenPurchases`
- `AvgTransactionValue`
- `ProductsPerTransaction`
- `QuantityPerTransaction`
- `ValuePerProduct`
- `PurchaseFrequency`
- `PriceSensitivity`
- `LoyaltyScore`

## 📈 Output Format

### Single Prediction Response
```json
{
    "prediction": 1,
    "segment": "Medium Value",
    "probabilities": [0.0, 1.0, 0.0],
    "probability_dict": {
        "0": 0.0,
        "1": 1.0,
        "2": 0.0
    },
    "confidence": 1.0,
    "timestamp": "2026-03-10T11:57:01.255282",
    "model_info": {
        "model_type": "Decision Tree",
        "features_used": 15
    }
}
```

### Batch Prediction Response
```json
{
    "results": [
        {
            "prediction": 1,
            "segment": "Medium Value",
            "probabilities": [0.0, 1.0, 0.0],
            "confidence": 1.0,
            "timestamp": "2026-03-10T11:57:01.255282",
            "model_info": {
                "model_type": "Decision Tree",
                "features_used": 15
            }
        }
    ],
    "batch_size": 1,
    "timestamp": "2026-03-10T11:57:01.255282"
}
```

## 🔧 Pipeline Components

### CustomerPredictionPipeline Class

#### Key Methods
- `load_artifacts()`: Load model and preprocessing artifacts
- `predict(data)`: Single customer prediction
- `predict_batch(data_list)`: Batch customer predictions
- `validate_input_data(data)`: Input data validation
- `create_derived_features(df)`: Feature engineering
- `select_features(df)`: Feature selection
- `scale_features(df)`: Feature scaling
- `get_feature_importance()`: Model feature importance
- `get_pipeline_info()`: Pipeline metadata

#### Features
- **Data Validation**: Comprehensive input validation with error handling
- **Feature Engineering**: Automatic creation of 8 derived features
- **Feature Selection**: Uses trained SelectKBest for optimal features
- **Scaling**: Applies trained scaler for consistent predictions
- **Error Handling**: Robust error handling with detailed logging
- **Batch Processing**: Efficient handling of multiple predictions

## 🚨 Error Handling

The pipeline includes comprehensive error handling for:
- Missing or invalid input data
- Model loading failures
- Feature engineering errors
- Prediction failures
- Data type mismatches

All errors are logged with detailed information for debugging.

## 📝 Logging

The pipeline uses Python's logging module with configurable levels:
- `INFO`: General operation information
- `WARNING`: Non-critical issues
- `ERROR`: Error conditions
- `DEBUG`: Detailed debugging information

## 🔍 Monitoring & Health Checks

### Pipeline Health
```python
pipeline = CustomerPredictionPipeline()
pipeline.load_artifacts()
info = pipeline.get_pipeline_info()
print(f"Pipeline loaded: {info['is_loaded']}")
print(f"Model type: {info['model_type']}")
```

### API Health Check
```bash
curl http://localhost:5000/health
```

## 🎯 Best Practices

1. **Data Quality**: Ensure input data is clean and validated
2. **Batch Processing**: Use batch predictions for multiple customers
3. **Error Handling**: Always wrap predictions in try-catch blocks
4. **Logging**: Monitor logs for performance and errors
5. **Model Updates**: Retrain models regularly with new data
6. **Feature Consistency**: Maintain consistent feature engineering

## 🔄 Integration Examples

### Integration with Web Applications
```python
from flask import Flask, request, jsonify
from src.prediction_pipeline import CustomerPredictionPipeline

app = Flask(__name__)
pipeline = CustomerPredictionPipeline()
pipeline.load_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        result = pipeline.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Integration with Data Processing Pipelines
```python
import pandas as pd
from src.prediction_pipeline import CustomerPredictionPipeline

# Load customer data
df = pd.read_csv('customers.csv')

# Initialize pipeline
pipeline = CustomerPredictionPipeline()
pipeline.load_artifacts()

# Batch prediction
customer_data = df.to_dict('records')
results = pipeline.predict_batch(customer_data)

# Add predictions to DataFrame
df['predicted_segment'] = [r['segment'] for r in results]
df['prediction_confidence'] = [r['confidence'] for r in results]
```

## 📚 Additional Resources

- **Model Training**: See `src/train_model.py` for training pipeline
- **Data Preprocessing**: See `src/data_preprocessing.py` for data cleaning
- **Feature Engineering**: See `src/feature_engineering.py` for feature creation
- **Streamlit App**: See `app/app.py` for web interface

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Failed**
   - Ensure model training was completed: `python src/train_model.py`
   - Check model artifacts exist in `artifacts/models/`

2. **Input Validation Errors**
   - Verify all required fields are present
   - Check data types match expected format
   - Ensure no missing values in critical fields

3. **Prediction Failures**
   - Check pipeline logs for detailed error information
   - Verify input data ranges are reasonable
   - Ensure model artifacts are compatible

4. **Performance Issues**
   - Use batch processing for multiple predictions
   - Monitor memory usage for large batches
   - Consider model optimization for production

## 📞 Support

For issues and questions:
1. Check the logs for detailed error information
2. Verify all prerequisites are met
3. Ensure model training was completed successfully
4. Review input data format and quality

---

**Note**: This pipeline is designed for production use with comprehensive error handling, logging, and monitoring capabilities.
