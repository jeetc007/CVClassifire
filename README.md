# Customer Classification ML Project

A complete machine learning project that classifies customers into value segments based on their purchasing behavior. The system uses transaction data to identify patterns and predict customer value categories using modern ML techniques and a beautiful Streamlit interface.

## 🎯 Project Overview

This project demonstrates a complete ML pipeline with:
- **Automated Data Processing**: Cleans and processes raw transaction data
- **Feature Engineering**: Creates meaningful customer-level features
- **Multi-Model Training**: Compares KNN, Decision Tree, and Random Forest models
- **Model Persistence**: Saves the best model for inference without retraining
- **Modern UI**: Interactive Streamlit dashboard with multiple pages
- **Customer Segmentation**: Classifies customers into Low, Medium, and High Value segments

## 📁 Project Structure

```
project_name/
│
├── data/
│   ├── raw/
│   │   └── online_retail_II.csv          # Your dataset goes here
│   │
│   └── processed/
│       └── processed_customer_data.csv   # Generated after preprocessing
│
├── artifacts/
│   ├── models/
│   │   ├── model.pkl                     # Trained model
│   │   ├── scaler.pkl                    # Feature scaler
│   │   ├── feature_selector.pkl          # Feature selector
│   │   ├── selected_features.pkl         # Selected feature names
│   │   └── model_metadata.pkl            # Model performance data
│   │
│   └── eda/
│       └── model_comparison.png          # Model comparison plot
│
├── notebooks/
│   └── training_pipeline.ipynb           # Exploratory analysis notebook
│
├── src/
│   ├── data_preprocessing.py             # Data cleaning and preprocessing
│   ├── feature_engineering.py            # Feature creation and selection
│   ├── train_model.py                    # Model training pipeline
│   └── predict.py                        # Inference pipeline
│
├── app/
│   └── app.py                            # Streamlit web application
│
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
└── .gitignore                            # Git ignore file
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Your Dataset

Place your CSV dataset in the `data/raw/` folder:

```bash
# Example: Move your dataset to the right location
cp your_dataset.csv data/raw/
```

**Expected Dataset Format:**
- Should contain transaction data with columns like:
  - `Customer ID`: Customer identifier
  - `Invoice`: Transaction identifier
  - `Quantity`: Number of items purchased
  - `Price`: Item price
  - `InvoiceDate`: Transaction date
  - `Country`: Customer country
  - `Description`: Product description

### 3. Run Training (One-time)

Train the models and save the best performing one:

```bash
python src/train_model.py
```

This will:
- Process your raw data
- Create customer-level features
- Train 3 different models
- Select and save the best model
- Generate performance visualizations

### 4. Run the Streamlit App

Start the web application:

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Streamlit App Features

The Streamlit application includes 4 main pages:

### 🏠 **Home Page**
- Project overview and description
- Model performance summary
- Dataset statistics
- Quick start guide

### 📊 **Data Analysis Page**
- Dataset summary and statistics
- Customer segment distribution
- Feature analysis with interactive charts
- Correlation matrix
- Data visualizations

### 🏆 **Model Performance Page**
- Model comparison table
- Performance charts (Accuracy, F1 Score)
- Best model details
- Feature importance visualization
- Training comparison plots

### 🔮 **Prediction Page**
- Interactive form for customer data input
- Real-time prediction results
- Prediction probabilities
- Input summary
- Model information

## 🤖 Machine Learning Pipeline

### Data Preprocessing
1. **Load Data**: Automatically reads CSV from `data/raw/`
2. **Clean Data**: Handles missing values, removes duplicates, filters invalid entries
3. **Feature Creation**: Creates customer-level features from transactions
4. **Encoding**: Converts categorical variables to numeric format

### Feature Engineering
- **Derived Features**: Average transaction value, purchase frequency, loyalty score
- **Feature Selection**: Statistical selection of top features
- **Scaling**: Standard scaling of numeric features

### Model Training
- **Algorithms**: KNN, Decision Tree, Random Forest
- **Evaluation**: Accuracy and F1 Score metrics
- **Selection**: Automatic selection of best performing model
- **Persistence**: Save model and preprocessing artifacts

### Inference
- **Loading**: Load saved model and preprocessing pipeline
- **Preprocessing**: Apply same transformations as training
- **Prediction**: Generate customer segment predictions
- **Interpretation**: Convert predictions to meaningful segment names

## 📈 Model Performance

The system evaluates models using:
- **Accuracy**: Overall prediction correctness
- **F1 Score**: Balance between precision and recall
- **Cross-validation**: Robust performance estimation

Expected performance metrics:
- Random Forest typically performs best (~90%+ accuracy)
- Decision Tree provides good interpretability
- KNN offers baseline performance

## 🔧 Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and preprocessing
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **streamlit**: Web application framework
- **plotly**: Interactive charts
- **joblib**: Model serialization

### Key Features
- **Modular Design**: Separate modules for each pipeline stage
- **Error Handling**: Comprehensive error handling and logging
- **Configurable**: Easy to modify parameters and settings
- **Scalable**: Handles datasets of various sizes
- **Reproducible**: Fixed random seeds for consistent results

## 🐛 Troubleshooting

### Common Issues

1. **"No CSV file found in data/raw folder"**
   - Solution: Place your dataset file in `data/raw/` directory

2. **"Model file not found"**
   - Solution: Run `python src/train_model.py` first

3. **"Error loading artifacts"**
   - Solution: Ensure training completed successfully and artifacts exist

4. **Streamlit app not loading**
   - Solution: Check all dependencies are installed with `pip install -r requirements.txt`

5. **Memory errors with large datasets**
   - Solution: Process data in chunks or use a machine with more RAM

### Performance Tips

- For large datasets, consider sampling during development
- Use GPU acceleration if available (not required for this project)
- Monitor memory usage during training
- Consider feature dimensionality reduction for very high-dimensional data

## 🔄 Workflow Summary

1. **Setup**: Install dependencies and add dataset
2. **Training**: Run `python src/train_model.py` (once)
3. **Inference**: Use `streamlit run app/app.py` for predictions
4. **Analysis**: Explore data and model performance in the app

## 📝 Notes

- The training process only needs to be run once per dataset
- The Streamlit app loads the saved model and doesn't retrain
- All preprocessing steps are saved and reused during inference
- The system automatically handles different dataset structures
- Model artifacts are saved in `artifacts/models/` for reuse

## 🤝 Contributing

Feel free to enhance the project with:
- Additional ML algorithms
- More sophisticated feature engineering
- Advanced visualizations
- Performance optimizations
- Additional evaluation metrics

## 📄 License

This project is open source and available under the MIT License.
