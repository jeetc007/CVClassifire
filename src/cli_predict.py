#!/usr/bin/env python3
"""
Command Line Interface for Customer Classification Prediction Pipeline

This script provides a CLI for making customer segmentation predictions
from the command line.
"""

import argparse
import json
import sys
from pathlib import Path
from prediction_pipeline import CustomerPredictionPipeline, create_sample_customer_data
import pandas as pd

def load_input_data(input_source: str) -> dict:
    """
    Load input data from various sources.
    
    Args:
        input_source: Path to JSON file, CSV file, or 'sample' for sample data
        
    Returns:
        dict: Input data for prediction
    """
    if input_source.lower() == 'sample':
        print("Using sample customer data...")
        return create_sample_customer_data()
    
    input_path = Path(input_source)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if input_path.suffix.lower() == '.json':
        print(f"Loading data from JSON file: {input_path}")
        with open(input_path, 'r') as f:
            return json.load(f)
    
    elif input_path.suffix.lower() == '.csv':
        print(f"Loading data from CSV file: {input_path}")
        df = pd.read_csv(input_path)
        
        # If CSV has multiple rows, use the first one
        if len(df) > 1:
            print(f"Warning: CSV has {len(df)} rows, using first row for single prediction")
        
        return df.iloc[0].to_dict()
    
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

def load_batch_data(input_source: str) -> list:
    """
    Load batch data from various sources.
    
    Args:
        input_source: Path to JSON file or CSV file
        
    Returns:
        list: List of customer data dictionaries
    """
    input_path = Path(input_source)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if input_path.suffix.lower() == '.json':
        print(f"Loading batch data from JSON file: {input_path}")
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of customer records")
        
        return data
    
    elif input_path.suffix.lower() == '.csv':
        print(f"Loading batch data from CSV file: {input_path}")
        df = pd.read_csv(input_path)
        return df.to_dict('records')
    
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

def save_results(results, output_file: str, format_type: str = 'json'):
    """
    Save prediction results to file.
    
    Args:
        results: Prediction results
        output_file: Output file path
        format_type: Output format ('json' or 'csv')
    """
    output_path = Path(output_file)
    
    if format_type.lower() == 'json':
        print(f"Saving results to JSON file: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif format_type.lower() == 'csv':
        print(f"Saving results to CSV file: {output_path}")
        
        if isinstance(results, dict):
            # Single prediction result
            df = pd.DataFrame([results])
        elif isinstance(results, list):
            # Batch prediction results
            df = pd.DataFrame(results)
        else:
            raise ValueError("Results must be a dictionary or list")
        
        df.to_csv(output_path, index=False)
    
    else:
        raise ValueError(f"Unsupported output format: {format_type}")

def print_prediction_result(result: dict, verbose: bool = False):
    """Print prediction result in a formatted way."""
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Customer Segment: {result['segment']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if verbose and 'probability_dict' in result:
        print("\nProbabilities:")
        for segment, prob in result['probability_dict'].items():
            segment_name = {0: 'Low Value', 1: 'Medium Value', 2: 'High Value'}[int(segment)]
            print(f"  {segment_name}: {prob:.3f}")
    
    if verbose and 'model_info' in result:
        print(f"\nModel Info:")
        print(f"  Type: {result['model_info']['model_type']}")
        print(f"  Features Used: {result['model_info']['features_used']}")
    
    print(f"Timestamp: {result['timestamp']}")
    print("="*50)

def print_batch_results(results: list, verbose: bool = False):
    """Print batch prediction results."""
    print(f"\n{'='*60}")
    print(f"BATCH PREDICTION RESULTS ({len(results)} customers)")
    print('='*60)
    
    for i, result in enumerate(results, 1):
        print(f"Customer {i}: {result['segment']} (confidence: {result['confidence']:.3f})")
        
        if verbose:
            print(f"  Prediction: {result['prediction']}")
            if 'probability_dict' in result:
                probs = result['probability_dict']
                max_prob = max(probs.values())
                print(f"  Max Probability: {max_prob:.3f}")
    
    print('='*60)

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Customer Classification Prediction CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction with sample data
  python cli_predict.py --single --input sample
  
  # Single prediction from JSON file
  python cli_predict.py --single --input customer_data.json
  
  # Batch prediction from CSV file
  python cli_predict.py --batch --input customers.csv
  
  # Save results to file
  python cli_predict.py --single --input sample --output results.json
  
  # Verbose output
  python cli_predict.py --single --input sample --verbose
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--single', action='store_true', 
                           help='Make single prediction')
    input_group.add_argument('--batch', action='store_true', 
                           help='Make batch prediction')
    
    # Data source
    parser.add_argument('--input', '-i', required=True,
                       help='Input data source (file path or "sample")')
    
    # Output options
    parser.add_argument('--output', '-o', 
                       help='Output file path (optional)')
    parser.add_argument('--format', '-f', choices=['json', 'csv'], default='json',
                       help='Output format (default: json)')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--model-dir', default='artifacts/models',
                       help='Model directory path (default: artifacts/models)')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        print("Initializing prediction pipeline...")
        pipeline = CustomerPredictionPipeline(model_dir=args.model_dir)
        
        # Load artifacts
        if not pipeline.load_artifacts():
            print("Error: Failed to load model artifacts", file=sys.stderr)
            sys.exit(1)
        
        print(f"Pipeline loaded successfully!")
        print(f"Model type: {pipeline.get_pipeline_info()['model_type']}")
        
        # Make prediction based on mode
        if args.single:
            # Single prediction
            print(f"\nMaking single prediction...")
            input_data = load_input_data(args.input)
            result = pipeline.predict(input_data)
            
            # Print result
            print_prediction_result(result, verbose=args.verbose)
            
            # Save result if requested
            if args.output:
                save_results(result, args.output, args.format)
                print(f"Results saved to: {args.output}")
        
        elif args.batch:
            # Batch prediction
            print(f"\nMaking batch prediction...")
            batch_data = load_batch_data(args.input)
            results = pipeline.predict_batch(batch_data)
            
            # Print results
            print_batch_results(results, verbose=args.verbose)
            
            # Save results if requested
            if args.output:
                save_results(results, args.output, args.format)
                print(f"Results saved to: {args.output}")
        
        # Show feature importance if verbose
        if args.verbose:
            print("\n" + "="*50)
            print("FEATURE IMPORTANCE")
            print("="*50)
            importance = pipeline.get_feature_importance()
            if importance:
                for i, (feature, score) in enumerate(list(importance.items())[:10]):
                    print(f"{i+1:2d}. {feature}: {score:.4f}")
            else:
                print("Feature importance not available")
            print("="*50)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
