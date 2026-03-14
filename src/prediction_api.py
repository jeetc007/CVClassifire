"""
Simple API wrapper for Customer Classification Prediction Pipeline

This module provides a simple interface for using the prediction pipeline
in web applications or other services.
"""

from flask import Flask, request, jsonify
from prediction_pipeline import CustomerPredictionPipeline, create_sample_customer_data
import logging
from datetime import datetime
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize prediction pipeline
pipeline = CustomerPredictionPipeline()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'pipeline_loaded': pipeline.is_loaded
    })

@app.route('/predict', methods=['POST'])
def predict_customer():
    """
    Predict customer segment for a single customer.
    
    Expected JSON payload:
    {
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
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        result = pipeline.predict(data)
        
        # Convert numpy types to Python types for JSON serialization
        if 'probabilities' in result:
            result['probabilities'] = [float(x) for x in result['probabilities']]
        
        if 'probability_dict' in result:
            result['probability_dict'] = {
                str(int(k)): float(v) for k, v in result['probability_dict'].items()
            }
        
        result['prediction'] = int(result['prediction'])
        result['confidence'] = float(result['confidence'])
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict customer segments for multiple customers.
    
    Expected JSON payload:
    [
        {
            "TotalAmount_Sum": 1500.0,
            "TotalAmount_Mean": 75.0,
            ...
        },
        {
            "TotalAmount_Sum": 5000.0,
            "TotalAmount_Mean": 250.0,
            ...
        }
    ]
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if not isinstance(data, list):
            return jsonify({'error': 'Data must be a list of customer records'}), 400
        
        # Make batch prediction
        results = pipeline.predict_batch(data)
        
        # Convert numpy types to Python types for JSON serialization
        for result in results:
            result['probabilities'] = [float(x) for x in result['probabilities']]
            result['prediction'] = int(result['prediction'])
            result['confidence'] = float(result['confidence'])
        
        return jsonify({
            'results': results,
            'batch_size': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/pipeline_info', methods=['GET'])
def get_pipeline_info():
    """Get information about the prediction pipeline."""
    return jsonify(pipeline.get_pipeline_info())

@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from the model."""
    try:
        importance = pipeline.get_feature_importance()
        if importance:
            # Convert numpy types to Python types
            importance_clean = {k: float(v) for k, v in importance.items()}
            return jsonify(importance_clean)
        else:
            return jsonify({'error': 'Feature importance not available'}), 404
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/sample_data', methods=['GET'])
def get_sample_data():
    """Get sample customer data for testing."""
    return jsonify(create_sample_customer_data())

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def init_app():
    """Initialize the application."""
    if not pipeline.load_artifacts():
        logger.error("Failed to load pipeline artifacts")
        return False
    logger.info("Pipeline loaded successfully")
    return True

if __name__ == '__main__':
    if init_app():
        print("Starting Customer Classification API...")
        print("Available endpoints:")
        print("  GET  /health            - Health check")
        print("  POST /predict           - Single prediction")
        print("  POST /predict_batch     - Batch prediction")
        print("  GET  /pipeline_info     - Pipeline information")
        print("  GET  /feature_importance - Feature importance")
        print("  GET  /sample_data       - Sample data for testing")
        print("\nAPI is running on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to initialize application")
