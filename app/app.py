import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys
import os

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set page configuration
st.set_page_config(
    page_title="Customer Classification ML Project",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-message {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CustomerClassificationApp:
    def __init__(self):
        self.predictor = None
        self.processed_data = None
        self.model_metadata = None
        
    def load_model_and_data(self):
        """Load the trained model and processed data using comprehensive pipeline"""
        try:
            # Import comprehensive prediction pipeline
            from src.prediction_pipeline import CustomerPredictionPipeline
            
            # Initialize pipeline and load artifacts
            self.predictor = CustomerPredictionPipeline()
            self.predictor.load_artifacts()
            
            # Get pipeline information
            self.pipeline_info = self.predictor.get_pipeline_info()
            
            # Load processed data
            processed_data_path = Path("data/processed/processed_customer_data.csv")
            if processed_data_path.exists():
                self.processed_data = pd.read_csv(processed_data_path)
            
            # Load model metadata
            metadata_path = Path("artifacts/models/model_metadata.pkl")
            if metadata_path.exists():
                self.model_metadata = joblib.load(metadata_path)

            return True
            
        except Exception as e:
            st.error(f"Error loading model or data: {e}")
            return False
    
    def render_home_page(self):
        """Render the home page"""
        st.markdown('<h1 class="main-header">🎯 Customer Classification System</h1>', unsafe_allow_html=True)
        
        # Project Overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h2 class="sub-header">Project Overview</h2>', unsafe_allow_html=True)
            st.markdown("""
            This machine learning project classifies customers into value segments based on their purchasing behavior. 
            The system uses transaction data to identify patterns and predict customer value categories.
            """)
            
            st.markdown("**Key Features:**")
            st.markdown("""
            - 📊 **Automated Data Processing**: Cleans and processes raw transaction data
            - 🤖 **Multi-Model Training**: Compares KNN, Decision Tree, and Random Forest models
            - 🎯 **Customer Segmentation**: Classifies customers into Low, Medium, and High Value segments
            - 📈 **Performance Analytics**: Comprehensive model evaluation and comparison
            - 🔮 **Real-time Prediction**: Interactive interface for customer classification
            """)
        
        with col2:
            if self.model_metadata:
                st.markdown('<h2 class="sub-header">Model Performance</h2>', unsafe_allow_html=True)
                
                best_model = self.model_metadata['model_name']
                best_scores = self.model_metadata['model_scores'][best_model]
                
                # Display metrics
                st.metric("Best Model", best_model)
                st.metric("Accuracy", f"{best_scores['accuracy']:.3f}")
                st.metric("F1 Score", f"{best_scores['f1_score']:.3f}")
                
                # Model comparison
                st.markdown("**All Models Performance:**")
                for model_name, scores in self.model_metadata['model_scores'].items():
                    if model_name == best_model:
                        st.markdown(f"✅ **{model_name}**: F1={scores['f1_score']:.3f}")
                    else:
                        st.markdown(f"📊 {model_name}: F1={scores['f1_score']:.3f}")
        
        # How it works
        st.markdown('<h2 class="sub-header">How It Works</h2>', unsafe_allow_html=True)
        
        steps = [
            "📁 Place your dataset in `data/raw/` folder",
            "🏃‍♂️ Run training: `python src/train_model.py`",
            "🚀 Start the app: `streamlit run app/app.py`",
            "🎯 Navigate through pages to analyze and predict"
        ]
        
        for i, step in enumerate(steps, 1):
            st.markdown(f"**{i}.** {step}")
        
        # Dataset Information
        if self.processed_data is not None:
            st.markdown('<h2 class="sub-header">Dataset Information</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Customers", len(self.processed_data))
            
            with col2:
                segment_dist = self.processed_data['CustomerSegment_Encoded'].value_counts()
                st.metric("Customer Segments", len(segment_dist))
            
            with col3:
                avg_amount = self.processed_data['TotalAmount_Sum'].mean()
                st.metric("Avg. Total Amount", f"${avg_amount:,.2f}")
            
            with col4:
                avg_transactions = self.processed_data['TransactionCount'].mean()
                st.metric("Avg. Transactions", f"{avg_transactions:.1f}")
    
    def render_data_analysis_page(self):
        """Render the comprehensive EDA page with 7 required visualizations"""
        st.markdown('<h1 class="main-header">📊 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
        
        if self.processed_data is None:
            st.warning("No processed data found. Please run the training pipeline first.")
            return
        
        # Load transaction data for product and time-based analysis
        try:
            transaction_data = pd.read_csv("data/processed/cleaned_transactions.csv")
        except:
            st.warning("Transaction data not found. Some visualizations may be limited.")
            transaction_data = None
        
        # CLV Tier Mapping (updated to match actual data)
        clv_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
        self.processed_data['CLV_Tier'] = self.processed_data['CustomerSegment_Encoded'].map(clv_mapping)
        
        # Graph 1: Histogram - Distribution of total customer spend
        st.markdown('<h2 class="sub-header">1. Distribution of Total Customer Spend</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            spend_bins = st.slider("Number of bins:", min_value=10, max_value=100, value=30, key="spend_bins")
            max_spend = st.slider("Maximum spend threshold:", min_value=1000, max_value=50000, value=10000, step=1000, key="max_spend")
        
        with col2:
            show_log = st.checkbox("Show log scale", value=False, key="spend_log")
        
        fig1 = px.histogram(
            self.processed_data[self.processed_data['TotalAmount_Sum'] <= max_spend],
            x='TotalAmount_Sum',
            nbins=spend_bins,
            title="Distribution of Customer Total Spend",
            labels={'TotalAmount_Sum': 'Total Spend ($)', 'count': 'Number of Customers'}
        )
        
        if show_log:
            fig1.update_layout(yaxis_type="log")
        
        fig1.add_vline(x=self.processed_data['TotalAmount_Sum'].mean(), line_dash="dash", 
                      annotation_text=f"Mean: ${self.processed_data['TotalAmount_Sum'].mean():.2f}")
        
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("""
        **💡 Business Insight:** The distribution shows customer spending patterns. A right-skewed distribution indicates 
        most customers spend relatively small amounts while a few high-value customers contribute significantly to revenue. 
        This helps identify the optimal spend thresholds for loyalty program tiers.
        """)
        
        st.markdown("---")
        
        # Graph 2: Scatter plot - Frequency vs. monetary value (CLV map)
        st.markdown('<h2 class="sub-header">2. Customer Lifetime Value (CLV) Map</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            freq_max = st.slider("Max frequency:", min_value=50, max_value=500, value=200, key="freq_max")
        with col2:
            show_trendline = st.checkbox("Show trend line", value=True, key="show_trend")
        
        # Filter data for better visualization
        plot_data = self.processed_data[
            (self.processed_data['TransactionCount'] <= freq_max) & 
            (self.processed_data['TotalAmount_Sum'] <= 20000)
        ].copy()
        
        fig2 = px.scatter(
            plot_data,
            x='TransactionCount',
            y='TotalAmount_Sum',
            color='CLV_Tier',
            size='UniqueProducts',
            hover_data=['CustomerID'],
            title="Frequency vs. Monetary Value (CLV Map)",
            labels={
                'TransactionCount': 'Purchase Frequency',
                'TotalAmount_Sum': 'Total Spend ($)',
                'UniqueProducts': 'Unique Products',
                'CLV_Tier': 'CLV Tier'
            },
            color_discrete_map={'Low': '#ff7f0e', 'Medium': '#2ca02c', 'High': '#1f77b4'}
        )
        
        if show_trendline:
            fig2.update_layout(
                shapes=[
                    dict(
                        type="line",
                        x0=0, y0=0, x1=freq_max, y1=20000,
                        line=dict(color="gray", width=1, dash="dash")
                    )
                ]
            )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        **💡 Business Insight:** This CLV map helps identify customer segments based on purchase frequency and monetary value. 
        High tier customers (top-right) show both high frequency and high spend, while Low tier customers (bottom-left) 
        have low frequency and spend. Targeted marketing strategies can be designed for each quadrant.
        """)
        
        st.markdown("---")
        
        # Graph 3: Bar chart - CLV tier breakdown by country
        st.markdown('<h2 class="sub-header">3. CLV Tier Distribution by Country</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            top_n_countries = st.slider("Show top N countries:", min_value=5, max_value=20, value=10, key="top_countries")
        with col2:
            show_percentage = st.checkbox("Show percentages", value=False, key="show_pct")
        
        # Create country mapping (reverse encoding)
        country_mapping = {1287: 'United Kingdom', 699726: 'Germany', 735: 'France', 
                          1243: 'Spain', 5349: 'Netherlands', 1: 'Australia'}
        self.processed_data['Country'] = self.processed_data['Country_Encoded'].map(country_mapping).fillna('Other')
        
        country_clv = self.processed_data.groupby(['Country', 'CLV_Tier']).size().reset_index(name='Count')
        top_countries = country_clv.groupby('Country')['Count'].sum().nlargest(top_n_countries).index
        country_clv_filtered = country_clv[country_clv['Country'].isin(top_countries)]
        
        if show_percentage:
            country_totals = country_clv_filtered.groupby('Country')['Count'].transform('sum')
            country_clv_filtered['Percentage'] = (country_clv_filtered['Count'] / country_totals * 100).round(1)
            y_col = 'Percentage'
            y_label = 'Percentage (%)'
        else:
            y_col = 'Count'
            y_label = 'Number of Customers'
        
        fig3 = px.bar(
            country_clv_filtered,
            x='Country',
            y=y_col,
            color='CLV_Tier',
            title=f"CLV Tier Distribution by Top {top_n_countries} Countries",
            labels={'Country': 'Country', y_col: y_label, 'CLV_Tier': 'CLV Tier'},
            color_discrete_map={'Low': '#ff7f0e', 'Medium': '#2ca02c', 'High': '#1f77b4'}
        )
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("""
        **💡 Business Insight:** Different countries show varying CLV tier distributions. Markets with higher High tier percentages 
        may be ripe for premium services expansion, while markets dominated by Low tier customers may need 
        acquisition and retention strategies.
        """)
        
        st.markdown("---")
        
        # Graph 4: Box plot - Average basket size by CLV tier
        st.markdown('<h2 class="sub-header">4. Average Basket Size by CLV Tier</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            basket_metric = st.selectbox("Basket metric:", ['TotalAmount_Mean', 'Quantity_Mean'], key="basket_metric")
        with col2:
            show_outliers = st.checkbox("Show outliers", value=False, key="show_outliers")
        
        metric_label = 'Average Transaction Value ($)' if basket_metric == 'TotalAmount_Mean' else 'Average Quantity per Transaction'
        
        fig4 = px.box(
            self.processed_data,
            x='CLV_Tier',
            y=basket_metric,
            title=f"Average Basket Size by CLV Tier",
            labels={'CLV_Tier': 'CLV Tier', basket_metric: metric_label},
            color='CLV_Tier',
            color_discrete_map={'Low': '#ff7f0e', 'Medium': '#2ca02c', 'High': '#1f77b4'}
        )
        
        if not show_outliers:
            fig4.update_traces(boxpoints=False)
        
        # Order tiers properly
        fig4.update_xaxes(categoryarray=['Low', 'Medium', 'High'])
        
        st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("""
        **💡 Business Insight:** High tier customers consistently show higher average basket sizes, indicating their willingness 
        to spend more per transaction. This insight can guide cross-selling and upselling strategies, 
        particularly for Medium tier customers who show potential for basket size improvement.
        """)
        
        st.markdown("---")
        
        # Graph 5: Bar chart - Top products purchased by High tier customers
        st.markdown('<h2 class="sub-header">5. Top Products Purchased by High Tier Customers</h2>', unsafe_allow_html=True)
        
        if transaction_data is not None:
            col1, col2 = st.columns(2)
            with col1:
                top_n_products = st.slider("Show top N products:", min_value=5, max_value=20, value=10, key="top_products")
            with col2:
                product_metric = st.selectbox("Metric:", ['Quantity', 'TotalAmount'], key="product_metric")
            
            # Get High tier customers
            high_tier_customers = self.processed_data[self.processed_data['CLV_Tier'] == 'High']['CustomerID'].tolist()
            
            # Filter transactions for High tier customers
            high_tier_transactions = transaction_data[transaction_data['Customer ID'].isin(high_tier_customers)]
            
            if not high_tier_transactions.empty:
                if product_metric == 'Quantity':
                    product_stats = high_tier_transactions.groupby('Description')['Quantity'].sum().nlargest(top_n_products).reset_index()
                    y_col = 'Quantity'
                    y_label = 'Total Quantity Purchased'
                else:
                    product_stats = high_tier_transactions.groupby('Description')['TotalAmount'].sum().nlargest(top_n_products).reset_index()
                    y_col = 'TotalAmount'
                    y_label = 'Total Revenue ($)'
                
                fig5 = px.bar(
                    product_stats,
                    x='Description',
                    y=y_col,
                    title=f"Top {top_n_products} Products by High Tier Customers",
                    labels={'Description': 'Product Description', y_col: y_label},
                    text_auto=True
                )
                fig5.update_xaxes(tickangle=45)
                fig5.update_traces(textposition='outside')
                st.plotly_chart(fig5, use_container_width=True)
                
                st.markdown("""
                **💡 Business Insight:** High tier customers prefer premium and high-value products. Understanding these preferences 
                helps in inventory optimization and targeted product recommendations for Medium tier customers 
                showing potential to move to High tier.
                """)
            else:
                st.warning("No High tier customer transaction data available.")
        else:
            st.warning("Transaction data not available for product analysis.")
        
        st.markdown("---")
        
        # Graph 6: Line chart - CLV tier composition over time (cohort)
        st.markdown('<h2 class="sub-header">6. CLV Tier Composition Over Time</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            time_period = st.selectbox("Time period:", ['Monthly', 'Quarterly'], key="time_period")
        with col2:
            show_cumulative = st.checkbox("Show cumulative", value=False, key="cumulative")
        
        # Create time period column
        if time_period == 'Monthly':
            self.processed_data['Period'] = self.processed_data['FirstPurchase_Year'].astype(str) + '-' + \
                                        self.processed_data['FirstPurchase_Month'].astype(str).str.zfill(2)
        else:
            quarter = ((self.processed_data['FirstPurchase_Month'] - 1) // 3) + 1
            self.processed_data['Period'] = self.processed_data['FirstPurchase_Year'].astype(str) + '-Q' + quarter.astype(str)
        
        time_clv = self.processed_data.groupby(['Period', 'CLV_Tier']).size().reset_index(name='Count')
        
        if show_cumulative:
            time_clv['Cumulative_Count'] = time_clv.groupby('CLV_Tier')['Count'].cumsum()
            y_col = 'Cumulative_Count'
            y_label = 'Cumulative Customer Count'
        else:
            y_col = 'Count'
            y_label = 'New Customer Count'
        
        fig6 = px.line(
            time_clv,
            x='Period',
            y=y_col,
            color='CLV_Tier',
            title=f"CLV Tier Composition Over Time ({time_period})",
            labels={'Period': time_period, y_col: y_label, 'CLV_Tier': 'CLV Tier'},
            color_discrete_map={'Low': '#ff7f0e', 'Medium': '#2ca02c', 'High': '#1f77b4'}
        )
        fig6.update_xaxes(tickangle=45)
        st.plotly_chart(fig6, use_container_width=True)
        
        st.markdown("""
        **💡 Business Insight:** Tracking CLV tier evolution over time reveals customer lifecycle patterns and the 
        effectiveness of retention strategies. Declining High tier acquisition may indicate the need for improved 
        premium service offerings.
        """)
        
        st.markdown("---")
        
        # Graph 7: Heatmap - Purchase pattern by day of week per tier
        st.markdown('<h2 class="sub-header">7. Purchase Patterns by Day of Week per CLV Tier</h2>', unsafe_allow_html=True)
        
        if transaction_data is not None:
            # Convert InvoiceDate to datetime and extract day of week
            transaction_data['InvoiceDate'] = pd.to_datetime(transaction_data['InvoiceDate'])
            transaction_data['DayOfWeek'] = transaction_data['InvoiceDate'].dt.day_name()
            
            # Map customer segments to transaction data
            customer_segments = self.processed_data[['CustomerID', 'CLV_Tier']].set_index('CustomerID')['CLV_Tier']
            transaction_data['CLV_Tier'] = transaction_data['Customer ID'].map(customer_segments)
            
            # Remove transactions without CLV tier info
            transaction_data_clean = transaction_data.dropna(subset=['CLV_Tier'])
            
            if not transaction_data_clean.empty:
                # Create purchase pattern data
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                tier_order = ['Low', 'Medium', 'High']
                
                purchase_pattern = transaction_data_clean.groupby(['CLV_Tier', 'DayOfWeek']).size().reset_index(name='TransactionCount')
                
                # Create pivot table for heatmap
                heatmap_data = purchase_pattern.pivot(index='CLV_Tier', columns='DayOfWeek', values='TransactionCount')
                heatmap_data = heatmap_data.reindex(columns=day_order, index=tier_order)
                
                # Fill missing days with 0
                heatmap_data = heatmap_data.fillna(0)
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    normalize = st.checkbox("Normalize by tier", value=True, key="normalize_heatmap")
                
                if normalize:
                    # Normalize by row (tier) to show percentage distribution
                    heatmap_normalized = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100
                    color_scale = 'Blues'
                    title_suffix = "(% of tier transactions)"
                else:
                    heatmap_normalized = heatmap_data
                    color_scale = 'Viridis'
                    title_suffix = "(Transaction Count)"
                
                fig7 = px.imshow(
                    heatmap_normalized,
                    title=f"Purchase Patterns by Day of Week per CLV Tier {title_suffix}",
                    labels={'x': 'Day of Week', 'y': 'CLV Tier', 'color': 'Transactions'},
                    color_continuous_scale=color_scale,
                    text_auto='.1f' if normalize else '.0f'
                )
                fig7.update_xaxes(side="bottom")
                st.plotly_chart(fig7, use_container_width=True)
                
                st.markdown("""
                **💡 Business Insight:** Different CLV tiers show distinct purchasing patterns throughout the week. 
                High tier customers may prefer weekend shopping while business customers might shop on weekdays. 
                These patterns can inform targeted marketing campaigns and promotional timing.
                """)
            else:
                st.warning("No transaction data available for purchase pattern analysis.")
        else:
            st.warning("Transaction data not available for purchase pattern analysis.")
    
    def render_model_performance_page(self):
        """Render the model performance page"""
        st.markdown('<h1 class="main-header">🏆 Model Performance</h1>', unsafe_allow_html=True)
        
        if self.model_metadata is None:
            st.warning("No model metadata found. Please run the training pipeline first.")
            return
        
        # Model Comparison
        st.markdown('<h2 class="sub-header">Model Comparison</h2>', unsafe_allow_html=True)
        
        # Create comparison table
        model_scores = self.model_metadata['model_scores']
        comparison_df = pd.DataFrame(model_scores).T
        comparison_df.columns = ['Accuracy', 'F1 Score']
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
        
        st.dataframe(comparison_df.style.background_gradient(subset=['F1 Score'], cmap='RdYlGn'))
        
        # Performance Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            fig_acc = px.bar(
                x=comparison_df.index,
                y=comparison_df['Accuracy'],
                title="Model Accuracy Comparison",
                labels={'x': 'Model', 'y': 'Accuracy'},
                color=comparison_df['Accuracy'],
                text_auto='.3f'
            )
            fig_acc.update_traces(textposition='outside')
            fig_acc.update_layout(showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            # F1 Score comparison
            fig_f1 = px.bar(
                x=comparison_df.index,
                y=comparison_df['F1 Score'],
                title="Model F1 Score Comparison",
                labels={'x': 'Model', 'y': 'F1 Score'},
                color=comparison_df['F1 Score'],
                text_auto='.3f'
            )
            fig_f1.update_traces(textposition='outside')
            fig_f1.update_layout(showlegend=False)
            st.plotly_chart(fig_f1, use_container_width=True)
        
        # Best Model Details
        st.markdown('<h2 class="sub-header">Best Model Details</h2>', unsafe_allow_html=True)
        
        best_model = self.model_metadata['model_name']
        best_scores = self.model_metadata['model_scores'][best_model]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"### 🥇 Best Model")
            st.markdown(f"## {best_model}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"### 📊 Accuracy")
            st.markdown(f"## {best_scores['accuracy']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 F1 Score")
            st.markdown(f"## {best_scores['f1_score']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature Importance (if available)
        if self.predictor:
            feature_importance = self.predictor.get_feature_importance()
            if feature_importance:
                st.markdown('<h2 class="sub-header">Feature Importance</h2>', unsafe_allow_html=True)
                
                # Convert to DataFrame for plotting
                importance_df = pd.DataFrame(
                    list(feature_importance.items()),
                    columns=['Feature', 'Importance']
                ).head(10)  # Top 10 features
                
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Feature Importances",
                    color='Importance',
                    text_auto='.3f'
                )
                fig_importance.update_traces(textposition='outside')
                fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # Load comparison plot if available
        comparison_plot_path = Path("artifacts/eda/model_comparison.png")
        if comparison_plot_path.exists():
            st.markdown('<h2 class="sub-header">Model Comparison Plot</h2>', unsafe_allow_html=True)
            st.image(str(comparison_plot_path), width='stretch')
    
    def render_prediction_page(self):
        """Render the prediction page"""
        st.markdown('<h1 class="main-header">🔮 Customer Prediction</h1>', unsafe_allow_html=True)
        
        if self.predictor is None:
            st.warning("Model not loaded. Please run the training pipeline first.")
            return
        
        st.markdown('<h2 class="sub-header">Enter Customer Data</h2>', unsafe_allow_html=True)
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Basic Information")
                
                total_amount_sum = st.number_input(
                    "Total Purchase Amount ($)",
                    min_value=0.0,
                    value=1000.0,
                    step=10.0
                )
                
                transaction_count = st.number_input(
                    "Number of Transactions",
                    min_value=1,
                    value=20,
                    step=1
                )
                
                unique_products = st.number_input(
                    "Number of Unique Products",
                    min_value=1,
                    value=15,
                    step=1
                )
                
                customer_tenure_days = st.number_input(
                    "Customer Tenure (Days)",
                    min_value=1,
                    value=365,
                    step=1
                )
                
                quantity_sum = st.number_input(
                    "Total Quantity Purchased",
                    min_value=1,
                    value=100,
                    step=1
                )
            
            with col2:
                st.subheader("Purchase Behavior")
                
                price_mean = st.number_input(
                    "Average Price per Item ($)",
                    min_value=0.1,
                    value=10.0,
                    step=0.5
                )
                
                first_purchase_year = st.number_input(
                    "First Purchase Year",
                    min_value=2000,
                    max_value=2024,
                    value=2023,
                    step=1
                )
                
                first_purchase_month = st.selectbox(
                    "First Purchase Month",
                    options=list(range(1, 13)),
                    index=0
                )
                
                last_purchase_year = st.number_input(
                    "Last Purchase Year",
                    min_value=2000,
                    max_value=2024,
                    value=2023,
                    step=1
                )
                
                last_purchase_month = st.selectbox(
                    "Last Purchase Month",
                    options=list(range(1, 13)),
                    index=11
                )
            
            # Submit button
            submit_button = st.form_submit_button("🎯 Predict Customer Segment")
            
            if submit_button:
                # Prepare input data
                input_data = {
                    'TotalAmount_Mean': total_amount_sum / transaction_count,
                    'TransactionCount': transaction_count,
                    'Quantity_Sum': quantity_sum,
                    'Quantity_Mean': quantity_sum / transaction_count,
                    'Price_Mean': price_mean,
                    'Country_Encoded': 100,  # Default encoding
                    'UniqueProducts': unique_products,
                    'CustomerTenureDays': customer_tenure_days,
                    'AvgDaysBetweenPurchases': customer_tenure_days / transaction_count,
                    'FirstPurchase_Year': first_purchase_year,
                    'FirstPurchase_Month': first_purchase_month,
                    'LastPurchase_Year': last_purchase_year,
                    'LastPurchase_Month': last_purchase_month
                }
                
                try:
                    # Debug: Show input values (optional)
                    if st.checkbox("Show Debug Info"):
                        st.write("Input values:", input_data)
                    
                    # Make prediction using comprehensive pipeline
                    with st.spinner("Making prediction..."):
                        result = self.predictor.predict(input_data)
                    
                    # Display results
                    st.markdown('<div class="success-message">', unsafe_allow_html=True)
                    st.markdown(f"## 🎯 Prediction Result")
                    st.markdown(f"### Customer Segment: **{result['segment']}**")
                    st.markdown(f"**Confidence:** {result['confidence']:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show probabilities if available
                    if result['probabilities']:
                        st.markdown('<h2 class="sub-header">Prediction Probabilities</h2>', unsafe_allow_html=True)
                        
                        # Create probability DataFrame
                        segment_names = ['Low Value', 'Medium Value', 'High Value']
                        prob_df = pd.DataFrame({
                            'Segment': segment_names,
                            'Probability': result['probabilities']
                        })
                        
                        fig_prob = px.bar(
                            prob_df,
                            x='Segment',
                            y='Probability',
                            title="Segment Probabilities",
                            color='Probability',
                            text_auto='.3f'
                        )
                        fig_prob.update_traces(textposition='outside')
                        fig_prob.update_layout(yaxis_range=[0, 1.1])
                        st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Show input summary
                    st.markdown('<h2 class="sub-header">Input Summary</h2>', unsafe_allow_html=True)
                    
                    summary_data = {
                        'Metric': [
                            'Total Amount', 'Transactions', 'Unique Products', 
                            'Customer Tenure', 'Avg Price', 'Total Quantity'
                        ],
                        'Value': [
                            f"${total_amount_sum:,.2f}",
                            str(transaction_count),
                            str(unique_products),
                            f"{customer_tenure_days} days",
                            f"${price_mean:.2f}",
                            str(quantity_sum)
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, hide_index=True)
                
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.markdown('<div class="warning-message">', unsafe_allow_html=True)
                    st.markdown("Please ensure all input values are valid and try again.")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Information
        if self.predictor:
            st.markdown('<h2 class="sub-header">Pipeline Information</h2>', unsafe_allow_html=True)
            
            pipeline_info = self.pipeline_info
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Details:**")
                st.write(f"- Type: {pipeline_info['model_type']}")
                st.write(f"- Total Features: {pipeline_info['total_features']}")
                st.write(f"- Selected Features: {pipeline_info['selected_features']}")
                st.write(f"- Feature Selector: {'✅' if pipeline_info['feature_selector_available'] else '❌'}")
                st.write(f"- Scaler: {'✅' if pipeline_info['scaler_available'] else '❌'}")
            
            with col2:
                st.markdown("**Segment Mapping:**")
                for segment_num, segment_name in pipeline_info['segment_mapping'].items():
                    st.write(f"- {segment_num}: {segment_name}")
                
                st.markdown("**Pipeline Status:**")
                st.write(f"- Loaded: {'✅' if pipeline_info['is_loaded'] else '❌'}")
                st.write(f"- Model Dir: {pipeline_info['model_dir']}")
        
        # Feature Importance
        if self.predictor:
            st.markdown('<h2 class="sub-header">Feature Importance</h2>', unsafe_allow_html=True)
            
            importance = self.predictor.get_feature_importance()
            if importance:
                # Create DataFrame for visualization
                importance_df = pd.DataFrame(
                    list(importance.items())[:10],  # Top 10 features
                    columns=['Feature', 'Importance']
                )
                
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Feature Importance",
                    color='Importance',
                    text_auto='.3f'
                )
                fig_importance.update_traces(textposition='outside')
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type")
    
    def render_model_training_page(self):
        """Render the model training page with algorithm metrics"""
        st.markdown('<h1 class="main-header">🚀 Model Training</h1>', unsafe_allow_html=True)
        
        st.markdown('<h2 class="sub-header">Train Classification Models</h2>', unsafe_allow_html=True)
        
        # Training button
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    # Import and run the training pipeline
                    import sys
                    import os
                    from pathlib import Path
                    
                    # Add src directory to path
                    src_path = Path(__file__).parent.parent / "src"
                    sys.path.append(str(src_path))
                    
                    # Import training module
                    from train_model import ModelTrainer
                    
                    # Initialize trainer
                    trainer = ModelTrainer()
                    
                    # Run training
                    training_results = trainer.train_all_models()
                    
                    # Store results in session state
                    st.session_state.training_results = training_results
                    st.session_state.training_completed = True
                    
                    st.success("✅ Model training completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {e}")
                    st.session_state.training_completed = False
        
        # Display results if training is completed
        if st.session_state.get('training_completed', False):
            results = st.session_state.get('training_results', {})
            
            if results:
                st.markdown('<h2 class="sub-header">📊 Model Performance Results</h2>', unsafe_allow_html=True)
                
                # Create algorithm cards
                algorithms = ['KNN', 'Decision Tree', 'Random Forest']
                
                for i, algorithm in enumerate(algorithms):
                    if algorithm in results:
                        metrics = results[algorithm]
                        
                        # Create a styled card for each algorithm
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 1.5rem; border-radius: 10px; 
                                    margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                            <h3 style="margin-bottom: 1rem;">🤖 {algorithm}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                label="📏 MAE", 
                                value=f"{metrics.get('mae', 0):.4f}",
                                delta=None
                            )
                        
                        with col2:
                            st.metric(
                                label="📐 RMSE", 
                                value=f"{metrics.get('rmse', 0):.4f}",
                                delta=None
                            )
                        
                        with col3:
                            st.metric(
                                label="📈 R²", 
                                value=f"{metrics.get('r2', 0):.4f}",
                                delta=f"{metrics.get('r2', 0):.4f}"
                            )
                        
                        with col4:
                            training_time = metrics.get('training_time', 0)
                            st.metric(
                                label="⏱️ Training Time", 
                                value=f"{training_time:.2f}s",
                                delta=None
                            )
                        
                        st.markdown("---")
                
                # Find best model
                best_model = max(results.keys(), key=lambda x: results[x].get('r2', 0))
                best_r2 = results[best_model].get('r2', 0)
                
                # Display best model
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                            color: white; padding: 1.5rem; border-radius: 10px; 
                            text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h2>🏆 Best Model: {best_model}</h2>
                    <p style="font-size: 1.2rem;">R² Score: {best_r2:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Model comparison chart
                st.markdown('<h2 class="sub-header">📈 Model Comparison</h2>', unsafe_allow_html=True)
                
                # Create comparison data
                comparison_data = []
                for algorithm, metrics in results.items():
                    comparison_data.append({
                        'Algorithm': algorithm,
                        'MAE': metrics.get('mae', 0),
                        'RMSE': metrics.get('rmse', 0),
                        'R²': metrics.get('r2', 0),
                        'Training Time (s)': metrics.get('training_time', 0)
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create comparison charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_r2 = px.bar(
                        comparison_df,
                        x='Algorithm',
                        y='R²',
                        title="R² Score Comparison",
                        color='R²',
                        text_auto='.3f'
                    )
                    fig_r2.update_traces(textposition='outside')
                    fig_r2.update_layout(showlegend=False)
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                with col2:
                    fig_time = px.bar(
                        comparison_df,
                        x='Algorithm',
                        y='Training Time (s)',
                        title="Training Time Comparison",
                        color='Training Time (s)',
                        text_auto='.2f'
                    )
                    fig_time.update_traces(textposition='outside')
                    fig_time.update_layout(showlegend=False)
                    st.plotly_chart(fig_time, use_container_width=True)
                
                # Detailed comparison table
                st.markdown('<h2 class="sub-header">📋 Detailed Metrics</h2>', unsafe_allow_html=True)
                st.dataframe(
                    comparison_df.style.background_gradient(subset=['R²'], cmap='RdYlGn')
                                   .background_gradient(subset=['MAE', 'RMSE'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
        
        else:
            # Instructions before training
            st.markdown("""
            <div style="background-color: #e3f2fd; border-left: 4px solid #2196f3; 
                        padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                <h3>📋 Training Instructions</h3>
                <p>Click the "Train Models" button above to train multiple classification algorithms 
                and compare their performance.</p>
                <p><strong>Algorithms to be trained:</strong></p>
                <ul>
                    <li>🤖 <strong>KNN</strong> - K-Nearest Neighbors</li>
                    <li>🌳 <strong>Decision Tree</strong> - Tree-based classifier</li>
                    <li>🌲 <strong>Random Forest</strong> - Ensemble method</li>
                </ul>
                <p><strong>Metrics displayed:</strong></p>
                <ul>
                    <li>📏 <strong>MAE</strong> - Mean Absolute Error</li>
                    <li>📐 <strong>RMSE</strong> - Root Mean Squared Error</li>
                    <li>📈 <strong>R²</strong> - R-squared (Coefficient of Determination)</li>
                    <li>⏱️ <strong>Training Time</strong> - Time taken to train the model</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Main application runner"""
        # Load model and data
        if not self.load_model_and_data():
            st.error("Failed to load model or data. Please run the training pipeline first.")
            st.markdown("""
            **To fix this issue:**
            1. Make sure your dataset is in `data/raw/` folder
            2. Run: `python src/train_model.py`
            3. Refresh this page
            """)
            return
        
        # Sidebar navigation
        st.sidebar.title("🗺️ Navigation")
        page = st.sidebar.radio(
            "Select a page:",
            ["🏠 Home", "📊 Data Analysis", "🏆 Model Performance", "🔮 Prediction", "🚀 Model Training"]
        )
        
        # Render selected page
        if page == "🏠 Home":
            self.render_home_page()
        elif page == "📊 Data Analysis":
            self.render_data_analysis_page()
        elif page == "🏆 Model Performance":
            self.render_model_performance_page()
        elif page == "🔮 Prediction":
            self.render_prediction_page()
        elif page == "🚀 Model Training":
            self.render_model_training_page()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Customer Classification System**")
        st.sidebar.markdown("Built with Streamlit & Scikit-learn")

# Main execution
if __name__ == "__main__":
    app = CustomerClassificationApp()
    app.run()
