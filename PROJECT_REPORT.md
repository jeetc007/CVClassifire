1. Introduction:
Customer segmentation plays a critical role in modern business strategies and customer relationship management. With the increasing volume of e-commerce transactions, understanding customer behavior has become essential for targeted marketing, personalized experiences, and maximizing revenue.
Machine learning techniques provide powerful tools for analyzing large transactional datasets and discovering hidden patterns in customer purchasing behavior. By learning from historical transaction data, these models can categorize customers into distinct value segments and help optimize marketing and retention efforts.
This project focuses on building a machine learning system capable of classifying customers into value segments (Low, Medium, and High Value) using historical purchasing data. The system analyzes transaction patterns and uses these patterns to predict a customer's segment based on engineered features such as purchase frequency, monetary value, and recency.
The project also includes an interactive web application that allows users to explore the dataset, view model performance, compare results, and generate real-time customer segment predictions.

2. Project Definition:
The goal of this project is to classify customers into different value categories based on historical transaction records. By applying data preprocessing and feature engineering techniques, meaningful customer-level features are extracted from raw transactional data. Three classification models are then trained to predict customer segments. The models are evaluated using performance metrics such as Accuracy and F1 Score to determine the most effective model. The final solution is implemented as a multi-page Streamlit web application that includes data visualization, model performance analysis, and a real-time prediction interface.

Problem Type: Classification
Dataset Name: Online Retail II Dataset (Processed into Customer-Level Data)
Dataset Characteristics:
● 4,148 unique customers (after preprocessing and outlier removal)
● 21 total engineered features (15 selected for the final model)
● 3 Target Classes: Low Value, Medium Value, High Value

3. Exploratory Data Analysis (EDA):
Exploratory Data Analysis (EDA) was conducted to understand purchasing patterns, segment distributions, and feature correlations within the customer dataset. Visualizations were created to examine how metrics like total spending, frequency, and recency vary across different customer segments. Through EDA, important insights were identified regarding high-value customer behaviors and feature importance. The following areas were visualized to analyze the dataset:

3.1 Customer Segment Distribution:
This chart shows the proportion of customers falling into the Low, Medium, and High Value segments. The visualization clarifies the class balance within the dataset, highlighting the distribution of the customer base.

![Customer Segment Distribution](file:///Users/jeetbhlodiyagmail.com/Downloads/CVClassifire/artifacts/eda/customer_classification_heatmap.png)

3.2 Feature Analysis and Distributions:
Histograms and box plots were used to display the distribution of key engineered features (e.g., Average Transaction Value, Purchase Frequency). These visualizations help identify the differences in purchasing behavior between the defined customer segments, showing that high-value customers typically exhibit distinct patterns in transaction frequency and monetary volume.

![Feature Distribution Analysis](file:///Users/jeetbhlodiyagmail.com/Downloads/CVClassifire/artifacts/eda/realistic_customer_heatmap.png)

3.3 Correlation Matrix:
This heatmap illustrates the correlation between different numerical features in the dataset. It helps identify which purchasing behaviors are strongly related to each other and safely guide the feature selection process by highlighting multicollinearity among variables.

![Correlation Matrix Heatmap](file:///Users/jeetbhlodiyagmail.com/Downloads/CVClassifire/artifacts/eda/heatmap_with_values.png)

4. Machine Learning Models:
In this project, three classification models from the scikit-learn library were utilized to predict customer value segments. These models learn patterns from historical customer features and estimate the target segment. The models implemented are K-Nearest Neighbors (KNN), Decision Tree Classifier, and Random Forest Classifier.

4.1 K-Nearest Neighbors (KNN):
KNN is an instance-based learning algorithm that classifies new data points based on the majority class of their 'k' nearest neighbors in the feature space. It calculates the distance (e.g., Euclidean distance) between the input customer data and all historical customer records. While simple and intuitive, KNN requires careful scaling of features and relies heavily on the choice of 'k'.

4.2 Decision Tree Classifier:
Decision Tree Classifier predicts class labels by splitting the dataset into subsets based on feature values. It creates a tree-like structure where internal nodes represent decision rules (e.g., if TotalAmount_Sum > X) and leaf nodes represent the predicted segment. Decision trees are highly interpretable and capable of capturing non-linear relationships. In this project, the Decision Tree model achieved exceptional performance, predicting the segments with 100% accuracy on the test data.

4.3 Random Forest Classifier:
Random Forest is an ensemble learning method that builds multiple decision trees during training and outputs the mode of the classes for classification. By using different subsets of data and features for each tree, Random Forest reduces the risk of overfitting associated with single decision trees and generally provides highly robust and accurate predictions.

5. Model Workflow:
The model workflow describes the end-to-end process of transforming raw transactional data into a robust machine learning system capable of real-time classification.

5.1 Data Collection:
The dataset used is based on online retail transactions, containing invoice details, quantities, prices, and customer IDs over a specific period. 

5.2 Data Preprocessing:
Raw transactional data was cleaned by handling missing values, removing duplicates, and filtering out invalid or canceled transactions (e.g., outlier removal, resolving garbage values).

5.3 Feature Engineering:
Since the goal is to classify customers, the transaction-level data was aggregated into customer-level data. Derived features such as average transaction value, total amount spent, and purchase frequency were generated. Feature selection techniques (SelectKBest) were applied to reduce the 21 created features down to the 15 most impactful ones. A StandardScaler was used to normalize the numerical data.

5.4 Train–Test Split:
The processed dataset was divided into a training set for model learning and a testing set for evaluating generalization on unseen data.

5.5 Model Training:
The three classification models (KNN, Decision Tree, Random Forest) were trained. During training, the models learned the boundaries separating Low, Medium, and High Value customers based on the 15 selected features.

5.6 Model Evaluation & Selection:
The models were evaluated using Accuracy and F1 Score. The Decision Tree Classifier emerged as the best-performing model, achieving 100% test accuracy and an F1 Score of 1.0. 

5.7 Model Persistence & Deployment:
To ensure instantaneous predictions, the best model and its preprocessing artifacts (scaler, feature selector) were saved as pickle files (`model.pkl`, `scaler.pkl`, etc.). The system was deployed via a Streamlit web application, allowing for inference directly from the saved artifacts without the overhead of retraining.

6. System Implementation and User Interface:
The customer classification system was deployed through an interactive Streamlit application. It offers a clean, robust interface for users to explore insights and perform real-time classification. The application consists of four main sections:

6.1 Home Page:
Provides an overview of the project, including the objective, dataset statistics, model performance summary, and a quick start guide. It introduces the user to the system's capabilities.

6.2 Data Analysis Page:
Displays interactive Exploratory Data Analysis. It offers dataset summaries, customer segment distributions, feature analysis charts, and a correlation matrix to provide deep insights into the customer base.

6.3 Model Performance Page:
Presents the evaluation results of the machine learning models. Users can view comparison tables, accuracy and F1 score charts, details of the best model (Decision Tree), and feature importance visualizations outlining which purchasing behaviors most heavily influence segment assignment.

![Model Performance Comparison](file:///Users/jeetbhlodiyagmail.com/Downloads/CVClassifire/artifacts/eda/detailed_model_comparison.png)

![Feature Importance Visualization](file:///Users/jeetbhlodiyagmail.com/Downloads/CVClassifire/artifacts/eda/feature_importance_perfect.png)

6.4 Prediction Page:
An interactive real-time inference interface. Users can input customer data metrics through a form. Using the pre-trained and loaded model artifacts (taking ~0.007 seconds per prediction), the system instantly outputs the predicted customer segment (Low, Medium, or High Value) alongside prediction probabilities.

Conclusion:
This project successfully developed a machine learning system to classify customers into value segments using historical retail transaction data. Through rigorous data preprocessing and aggregation, transactional records were converted into meaningful customer features. Three classification models were trained and evaluated, with the Decision Tree Classifier selected as the optimal model due to its 100% accuracy. The entire pipeline, including an interactive web application built with Streamlit, allows businesses to efficiently explore customer data, evaluate model metrics, and categorize new customers in real-time. This system demonstrates the practical value of machine learning in enabling data-driven marketing and customer relationship management.
