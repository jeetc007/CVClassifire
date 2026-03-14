#!/usr/bin/env python3
"""
Test script for Customer Classification Prediction Pipeline

This script demonstrates various capabilities of the prediction pipeline
including single predictions, batch processing, and error handling.
"""

import json
import pandas as pd
from src.prediction_pipeline import CustomerPredictionPipeline, create_sample_customer_data

def test_single_prediction():
    """Test single customer prediction."""
    print("=" * 60)
    print("TEST 1: Single Customer Prediction")
    print("=" * 60)
    
    pipeline = CustomerPredictionPipeline()
    
    if not pipeline.load_artifacts():
        print("❌ Failed to load pipeline artifacts")
        return False
    
    # Test with sample data
    sample_data = create_sample_customer_data()
    print(f"Input data: {json.dumps(sample_data, indent=2)}")
    
    try:
        result = pipeline.predict(sample_data)
        print(f"✅ Prediction successful!")
        print(f"   Segment: {result['segment']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Model: {result['model_info']['model_type']}")
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def test_batch_prediction():
    """Test batch customer prediction."""
    print("\n" + "=" * 60)
    print("TEST 2: Batch Customer Prediction")
    print("=" * 60)
    
    pipeline = CustomerPredictionPipeline()
    
    if not pipeline.load_artifacts():
        print("❌ Failed to load pipeline artifacts")
        return False
    
    # Create test batch data
    batch_data = [
        create_sample_customer_data(),  # Medium value
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
        },  # High value
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
        }   # Low value
    ]
    
    print(f"Batch size: {len(batch_data)} customers")
    
    try:
        results = pipeline.predict_batch(batch_data)
        print(f"✅ Batch prediction successful!")
        
        for i, result in enumerate(results, 1):
            print(f"   Customer {i}: {result['segment']} (confidence: {result['confidence']:.3f})")
        
        return True
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        return False

def test_input_validation():
    """Test input validation and error handling."""
    print("\n" + "=" * 60)
    print("TEST 3: Input Validation & Error Handling")
    print("=" * 60)
    
    pipeline = CustomerPredictionPipeline()
    
    if not pipeline.load_artifacts():
        print("❌ Failed to load pipeline artifacts")
        return False
    
    # Test cases for validation
    test_cases = [
        {
            'name': 'Missing required field',
            'data': {
                'TotalAmount_Sum': 1000.0,
                'TransactionCount': 10,
                # Missing other required fields
            },
            'should_fail': True
        },
        {
            'name': 'Negative values in critical fields',
            'data': {
                'TotalAmount_Sum': -100.0,
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
            },
            'should_fail': False  # Should work but with warnings
        },
        {
            'name': 'Valid data',
            'data': create_sample_customer_data(),
            'should_fail': False
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        
        try:
            result = pipeline.predict(test_case['data'])
            
            if test_case['should_fail']:
                print(f"❌ Expected failure but prediction succeeded")
            else:
                print(f"✅ Prediction succeeded as expected")
                print(f"   Result: {result['segment']}")
                passed += 1
                
        except Exception as e:
            if test_case['should_fail']:
                print(f"✅ Failed as expected: {e}")
                passed += 1
            else:
                print(f"❌ Unexpected failure: {e}")
    
    print(f"\nValidation tests passed: {passed}/{total}")
    return passed == total

def test_pipeline_info():
    """Test pipeline information and feature importance."""
    print("\n" + "=" * 60)
    print("TEST 4: Pipeline Information & Feature Importance")
    print("=" * 60)
    
    pipeline = CustomerPredictionPipeline()
    
    if not pipeline.load_artifacts():
        print("❌ Failed to load pipeline artifacts")
        return False
    
    # Get pipeline info
    info = pipeline.get_pipeline_info()
    print("Pipeline Information:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Get feature importance
    print("\nFeature Importance:")
    importance = pipeline.get_feature_importance()
    if importance:
        for i, (feature, score) in enumerate(list(importance.items())[:5]):
            print(f"   {i+1}. {feature}: {score:.4f}")
        print(f"   ... and {len(importance)-5} more features")
    else:
        print("   Feature importance not available")
    
    print("✅ Pipeline information retrieved successfully!")
    return True

def test_dataframe_input():
    """Test prediction with DataFrame input."""
    print("\n" + "=" * 60)
    print("TEST 5: DataFrame Input Handling")
    print("=" * 60)
    
    pipeline = CustomerPredictionPipeline()
    
    if not pipeline.load_artifacts():
        print("❌ Failed to load pipeline artifacts")
        return False
    
    # Create DataFrame
    data = [
        create_sample_customer_data(),
        {
            'TotalAmount_Sum': 3000.0,
            'TotalAmount_Mean': 150.0,
            'TransactionCount': 20,
            'Quantity_Sum': 300,
            'Quantity_Mean': 15.0,
            'Price_Mean': 10.0,
            'Country_Encoded': 100,
            'UniqueProducts': 25,
            'CustomerTenureDays': 200,
            'FirstPurchase_Year': 2022,
            'FirstPurchase_Month': 6,
            'LastPurchase_Year': 2023,
            'LastPurchase_Month': 6
        }
    ]
    
    df = pd.DataFrame(data)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    
    try:
        # Test single row prediction (convert Series to dict)
        single_result = pipeline.predict(df.iloc[0].to_dict())
        print(f"✅ Single DataFrame row prediction: {single_result['segment']}")
        
        # Test batch DataFrame prediction
        batch_results = pipeline.predict_batch(df)
        print(f"✅ Batch DataFrame prediction: {len(batch_results)} results")
        
        for i, result in enumerate(batch_results):
            print(f"   Row {i+1}: {result['segment']}")
        
        return True
    except Exception as e:
        print(f"❌ DataFrame prediction failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🧪 CUSTOMER CLASSIFICATION PREDICTION PIPELINE TESTS")
    print("=" * 60)
    
    tests = [
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Input Validation", test_input_validation),
        ("Pipeline Information", test_pipeline_info),
        ("DataFrame Input", test_dataframe_input)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()
