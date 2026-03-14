import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.artifacts_path = Path("artifacts/models")
        self.eda_path = Path("artifacts/eda")
        
    def prepare_data(self):
        """Prepare data for training"""
        print("Preparing data for training...")
        
        # Load and preprocess data
        from data_preprocessing import DataPreprocessor
        from feature_engineering import FeatureEngineer
        
        preprocessor = DataPreprocessor()
        processed_df, _ = preprocessor.preprocess_pipeline()
        
        # Apply feature engineering
        fe = FeatureEngineer()
        X, y = fe.feature_engineering_pipeline(processed_df, fit_scaler=True)
        
        print(f"Data prepared successfully. Features: {X.shape[1]}, Samples: {X.shape[0]}")
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print("Splitting data into train and test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """Initialize the three models"""
        print("Initializing models...")
        
        self.models = {
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        print("Models initialized:")
        for name, model in self.models.items():
            print(f"  - {name}: {model}")
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            print(f"{name} training completed")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and store scores"""
        print("Evaluating models...")
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Store scores
            self.model_scores[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'model': model
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Print detailed classification report
            print(f"\nClassification Report for {name}:")
            print(classification_report(y_test, y_pred))
    
    def select_best_model(self):
        """Select the best performing model based on F1 score"""
        print("\nSelecting best model...")
        
        # Find model with highest F1 score
        best_name = max(self.model_scores.keys(), 
                       key=lambda x: self.model_scores[x]['f1_score'])
        
        self.best_model = self.model_scores[best_name]['model']
        self.best_model_name = best_name
        
        print(f"\nBest model: {best_name}")
        print(f"Best F1 Score: {self.model_scores[best_name]['f1_score']:.4f}")
        print(f"Best Accuracy: {self.model_scores[best_name]['accuracy']:.4f}")
        
        return self.best_model_name, self.model_scores[best_name]
    
    def save_model(self):
        """Save the best model"""
        print(f"Saving best model: {self.best_model_name}")
        
        # Create artifacts directory
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Save the best model
        model_path = self.artifacts_path / "model.pkl"
        joblib.dump(self.best_model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save model name and scores
        metadata = {
            'model_name': self.best_model_name,
            'model_scores': {k: {'accuracy': v['accuracy'], 'f1_score': v['f1_score']} 
                           for k, v in self.model_scores.items()}
        }
        
        metadata_path = self.artifacts_path / "model_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        print(f"Model metadata saved to: {metadata_path}")
        
        return model_path
    
    def create_comparison_plots(self):
        """Create model comparison plots"""
        print("Creating model comparison plots...")
        
        # Create EDA directory
        self.eda_path.mkdir(parents=True, exist_ok=True)
        
        # Extract scores for plotting
        models = list(self.model_scores.keys())
        accuracies = [self.model_scores[model]['accuracy'] for model in models]
        f1_scores = [self.model_scores[model]['f1_score'] for model in models]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        bars1 = ax1.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # F1 Score comparison
        bars2 = ax2.bar(models, f1_scores, color=['skyblue', 'lightgreen', 'salmon'])
        ax2.set_title('Model F1 Score Comparison')
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0, 1)
        # Add value labels on bars
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.eda_path / "model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison plot saved to: {plot_path}")
        
        return plot_path
    
    def training_pipeline(self):
        """Complete training pipeline"""
        print("=" * 60)
        print("STARTING MODEL TRAINING PIPELINE")
        print("=" * 60)
        
        # Prepare data
        X, y = self.prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Initialize models
        self.initialize_models()
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Select best model
        best_name, best_scores = self.select_best_model()
        
        # Save model
        model_path = self.save_model()
        
        # Create comparison plots
        plot_path = self.create_comparison_plots()
        
        print("\n" + "=" * 60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Best Model: {best_name}")
        print(f"Best Model F1 Score: {best_scores['f1_score']:.4f}")
        print(f"Best Model Accuracy: {best_scores['accuracy']:.4f}")
        print(f"Model saved to: {model_path}")
        print(f"Comparison plot saved to: {plot_path}")
        
        return best_name, best_scores

if __name__ == "__main__":
    # Run the complete training pipeline
    trainer = ModelTrainer()
    best_model_name, best_scores = trainer.training_pipeline()
    
    print(f"\nTraining completed! Best model: {best_model_name}")
    print("You can now run the Streamlit app with: streamlit run app/app.py")
