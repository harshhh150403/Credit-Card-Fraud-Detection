# credit_card_fraud_detection/app.py
"""
Streamlit Dashboard for Credit Card Fraud Detection System
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.config import DATASET_FILENAME
from src.utils import load_model, load_results


class FraudDetectionDashboard:
    """Streamlit dashboard for fraud detection results"""
    
    def __init__(self):
        self.results = None
        self.model = None
        self.threshold_optimizer = None
        self.data_loader = None
        
    def load_artifacts(self):
        """Load model and results"""
        try:
            self.results = load_results("pipeline_results.json")
            st.success("✅ Results loaded successfully")
        except:
            st.warning("⚠️ Results not found. Please run the pipeline first.")
            self.results = None
            
        try:
            self.model = load_model("best_fraud_model.pkl")
            st.success("✅ Model loaded successfully")
        except:
            st.warning("⚠️ Model not found. Please run the pipeline first.")
            self.model = None
            
        try:
            self.threshold_optimizer = load_model("threshold_optimizer.pkl")
            st.success("✅ Threshold optimizer loaded successfully")
        except:
            st.warning("⚠️ Threshold optimizer not found")
            self.threshold_optimizer = None
            
        self.data_loader = DataLoader(DATASET_FILENAME)
        
    def sidebar(self):
        """Create sidebar with navigation"""
        st.sidebar.title("🔍 Fraud Detection Dashboard")
        st.sidebar.markdown("---")
        
        page = st.sidebar.radio(
            "Navigation",
            ["📊 Project Overview", "📈 Model Performance", "🔧 Feature Analysis", 
             "🎯 Live Prediction", "📋 Data Explorer"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            "**About:**\n"
            "Credit Card Fraud Detection System using XGBoost with SMOTE and threshold optimization."
        )
        
        return page
    
    def project_overview(self):
        """Project overview page"""
        st.title("💰 Credit Card Fraud Detection System")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if self.results:
                st.metric(
                    "Final F1-Score", 
                    f"{self.results['final_metrics']['f1_score']:.4f}"
                )
        
        with col2:
            if self.results:
                st.metric(
                    "Optimal Threshold", 
                    f"{self.results['optimal_threshold']:.3f}"
                )
        
        with col3:
            if self.results:
                st.metric(
                    "Fraud Detection Rate", 
                    f"{self.results['final_metrics']['recall']:.4f}"
                )
        
        st.markdown("""
        ### 🎯 Project Overview
        
        This system detects fraudulent credit card transactions using machine learning with the following features:
        
        - **Advanced Feature Engineering**: Domain-specific features for fraud patterns
        - **Class Imbalance Handling**: SMOTE oversampling for fraud cases
        - **Hyperparameter Optimization**: Bayesian optimization for XGBoost
        - **Threshold Optimization**: Dynamic threshold tuning for business needs
        - **Production Ready**: Modular pipeline architecture
        
        ### 📊 Dataset Summary
        """)
        
        try:
            data = self.data_loader.load_cleaned_data()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", f"{len(data):,}")
            
            with col2:
                fraud_count = data['isFraud'].sum()
                st.metric("Fraudulent Transactions", f"{fraud_count:,}")
            
            with col3:
                fraud_rate = (data['isFraud'].mean() * 100)
                st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
            
            with col4:
                st.metric("Data Features", f"{len(data.columns)}")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    def model_performance(self):
        """Model performance visualization"""
        st.title("📈 Model Performance Analysis")
        st.markdown("---")
        
        if not self.results:
            st.warning("Please run the pipeline first to see performance results")
            return
        
        # Key metrics comparison
        st.subheader("📊 Performance Metrics Comparison")
        
        col1, col2, col3, col4 = st.columns(4)
        
        initial = self.results['initial_metrics']
        final = self.results['final_metrics']
        
        with col1:
            delta_f1 = final['f1_score'] - initial['f1_score']
            st.metric("F1-Score", f"{final['f1_score']:.4f}", f"{delta_f1:+.4f}")
        
        with col2:
            delta_precision = final['precision'] - initial['precision']
            st.metric("Precision", f"{final['precision']:.4f}", f"{delta_precision:+.4f}")
        
        with col3:
            delta_recall = final['recall'] - initial['recall']
            st.metric("Recall", f"{final['recall']:.4f}", f"{delta_recall:+.4f}")
        
        with col4:
            if 'roc_auc' in final:
                st.metric("ROC AUC", f"{final['roc_auc']:.4f}")
        
        # Metrics comparison chart
        st.subheader("📈 Metrics Before vs After Threshold Optimization")
        
        metrics_data = {
            'Metric': ['F1-Score', 'Precision', 'Recall'],
            'Before': [initial['f1_score'], initial['precision'], initial['recall']],
            'After': [final['f1_score'], final['precision'], final['recall']]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Before Optimization',
            x=df_metrics['Metric'],
            y=df_metrics['Before'],
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='After Optimization',
            x=df_metrics['Metric'],
            y=df_metrics['After'],
            marker_color='royalblue'
        ))
        
        fig.update_layout(
            title="Performance Metrics Comparison",
            xaxis_title="Metrics",
            yaxis_title="Score",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix
        st.subheader("🎯 Confusion Matrix")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before Threshold Optimization**")
            cm_initial = initial['confusion_matrix']
            self._display_confusion_matrix(cm_initial)
        
        with col2:
            st.markdown("**After Threshold Optimization**")
            cm_final = final['confusion_matrix']
            self._display_confusion_matrix(cm_final)
        
        # Detailed metrics
        st.subheader("📋 Detailed Performance Metrics")
        
        detailed_data = {
            'Metric': [
                'F1-Score', 'Precision', 'Recall', 'Accuracy',
                'False Positive Rate', 'False Negative Rate',
                'True Positive Rate', 'True Negative Rate'
            ],
            'Before': [
                initial['f1_score'], initial['precision'], initial['recall'],
                initial['accuracy'], initial['false_positive_rate'],
                initial['false_negative_rate'], initial['true_positive_rate'],
                initial['true_negative_rate']
            ],
            'After': [
                final['f1_score'], final['precision'], final['recall'],
                final['accuracy'], final['false_positive_rate'],
                final['false_negative_rate'], final['true_positive_rate'],
                final['true_negative_rate']
            ]
        }
        
        df_detailed = pd.DataFrame(detailed_data)
        st.dataframe(df_detailed, use_container_width=True)
    
    def _display_confusion_matrix(self, cm):
        """Display confusion matrix as a formatted table"""
        df_cm = pd.DataFrame({
            'Actual Legitimate': [cm['true_negatives'], cm['false_negatives']],
            'Actual Fraud': [cm['false_positives'], cm['true_positives']]
        }, index=['Predicted Legitimate', 'Predicted Fraud'])
        
        st.dataframe(df_cm.style.format('{:,.0f}'), use_container_width=True)
        
        # Calculate percentages
        total = cm['true_negatives'] + cm['false_positives'] + cm['false_negatives'] + cm['true_positives']
        st.caption(f"Total: {total:,} transactions")
    
    def feature_analysis(self):
        """Feature importance and analysis"""
        st.title("🔧 Feature Analysis")
        st.markdown("---")
        
        if not self.model:
            st.warning("Model not available for feature analysis")
            return
        
        try:
            # Get feature importance from XGBoost model
            if hasattr(self.model.named_steps['model'], 'feature_importances_'):
                feature_importances = self.model.named_steps['model'].feature_importances_
                
                # Get feature names from preprocessor
                feature_names = self.model.named_steps['preprocessor'].get_feature_names()
                
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importances
                }).sort_values('importance', ascending=False)
                
                st.subheader("📊 Feature Importance")
                
                # Display top features
                fig = px.bar(
                    importance_df.head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 15 Most Important Features'
                )
                
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Show full feature importance table
                with st.expander("View All Feature Importances"):
                    st.dataframe(importance_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in feature analysis: {str(e)}")
        
        # Feature engineering insights
        st.subheader("🎯 Key Fraud Indicators")
        
        fraud_insights = [
            "💡 **Account Emptying**: Fraudsters often transfer entire account balances",
            "💡 **Transaction Types**: Fraud mainly occurs in TRANSFER and CASH_OUT transactions", 
            "💡 **Amount Patterns**: Fraud transactions are typically larger than legitimate ones",
            "💡 **Balance Changes**: Negative balance changes in origin accounts indicate fraud",
            "💡 **Zero Balances**: Many fraud accounts start with zero or low balances"
        ]
        
        for insight in fraud_insights:
            st.markdown(insight)
    
    def live_prediction(self):
        """Live transaction prediction interface"""
        st.title("🎯 Live Fraud Prediction")
        st.markdown("---")
        
        if not self.model:
            st.warning("Model not available for predictions")
            return
        
        st.markdown("""
        ### Enter Transaction Details
        Provide the transaction information to check if it's potentially fraudulent.
        """)
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                transaction_type = st.selectbox(
                    "Transaction Type",
                    ["TRANSFER", "CASH_OUT"]
                )
                
                amount = st.number_input(
                    "Amount",
                    min_value=0.0,
                    max_value=10000000.0,
                    value=1000.0,
                    step=100.0
                )
                
                oldbalance_org = st.number_input(
                    "Origin Old Balance",
                    min_value=0.0,
                    max_value=100000000.0,
                    value=5000.0,
                    step=100.0
                )
                
                newbalance_orig = st.number_input(
                    "Origin New Balance",
                    min_value=0.0,
                    max_value=100000000.0,
                    value=4000.0,
                    step=100.0
                )
            
            with col2:
                oldbalance_dest = st.number_input(
                    "Destination Old Balance",
                    min_value=0.0,
                    max_value=100000000.0,
                    value=1000.0,
                    step=100.0
                )
                
                newbalance_dest = st.number_input(
                    "Destination New Balance", 
                    min_value=0.0,
                    max_value=100000000.0,
                    value=2000.0,
                    step=100.0
                )
                
                step = st.slider(
                    "Time Step (Hour)",
                    min_value=1,
                    max_value=743,
                    value=100
                )
            
            submitted = st.form_submit_button("🔍 Predict Fraud")
            
            if submitted:
                # Create input data
                input_data = pd.DataFrame({
                    'step': [step],
                    'type': [transaction_type],
                    'amount': [amount],
                    'oldbalanceOrg': [oldbalance_org],
                    'newbalanceOrig': [newbalance_orig],
                    'oldbalanceDest': [oldbalance_dest],
                    'newbalanceDest': [newbalance_dest],
                    'nameOrig': ['C123456789'],  # Dummy value
                    'nameDest': ['M987654321']   # Dummy value
                })
                
                try:
                    # Get prediction
                    prediction = self.model.predict(input_data)[0]
                    probability = self.model.predict_proba(input_data)[0][1]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.error(f"🚨 **FRAUD DETECTED**")
                        else:
                            st.success(f"✅ **LEGITIMATE TRANSACTION**")
                    
                    with col2:
                        st.metric("Fraud Probability", f"{probability:.4f}")
                    
                    # Show feature values
                    st.subheader("Transaction Analysis")
                    
                    # Calculate engineered features
                    amount_balance_ratio = amount / (oldbalance_org + 1e-9)
                    balance_change = newbalance_orig - oldbalance_org
                    
                    analysis_data = {
                        'Feature': [
                            'Amount to Balance Ratio',
                            'Balance Change', 
                            'Transaction Type',
                            'Amount'
                        ],
                        'Value': [
                            f"{amount_balance_ratio:.4f}",
                            f"${balance_change:,.2f}",
                            transaction_type,
                            f"${amount:,.2f}"
                        ],
                        'Risk Indicator': [
                            '⚠️ High Risk' if amount_balance_ratio > 0.9 else '✅ Normal',
                            '⚠️ High Risk' if balance_change < -amount * 0.1 else '✅ Normal', 
                            '⚠️ High Risk' if transaction_type in ['TRANSFER', 'CASH_OUT'] else '✅ Normal',
                            '⚠️ High Risk' if amount > 100000 else '✅ Normal'
                        ]
                    }
                    
                    st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    def data_explorer(self):
        """Data exploration interface"""
        st.title("📋 Data Explorer")
        st.markdown("---")
        
        try:
            data = self.data_loader.load_cleaned_data()
            
            st.subheader("Dataset Overview")
            
            # Basic statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Transactions", f"{len(data):,}")
                st.metric("Fraudulent Transactions", f"{data['isFraud'].sum():,}")
            
            with col2:
                fraud_rate = data['isFraud'].mean() * 100
                st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
                st.metric("Features", f"{len(data.columns)}")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(1000), use_container_width=True)
            
            # Column information
            st.subheader("Column Information")
            
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes,
                'Non-Null Count': data.count(),
                'Null Count': data.isnull().sum()
            })
            
            st.dataframe(col_info, use_container_width=True)
            
            # Transaction type distribution
            st.subheader("Transaction Type Distribution")
            
            type_counts = data['type'].value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Transaction Types"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Fraud by transaction type
            st.subheader("Fraud Distribution by Transaction Type")
            
            fraud_by_type = data.groupby('type')['isFraud'].mean() * 100
            fig = px.bar(
                x=fraud_by_type.index,
                y=fraud_by_type.values,
                title="Fraud Rate by Transaction Type (%)",
                labels={'x': 'Transaction Type', 'y': 'Fraud Rate (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    def run(self):
        """Run the dashboard"""
        st.set_page_config(
            page_title="Fraud Detection Dashboard",
            page_icon="💰",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Load artifacts
        with st.spinner("Loading model and results..."):
            self.load_artifacts()
        
        # Navigation
        page = self.sidebar()
        
        # Page routing
        if page == "📊 Project Overview":
            self.project_overview()
        elif page == "📈 Model Performance":
            self.model_performance()
        elif page == "🔧 Feature Analysis":
            self.feature_analysis()
        elif page == "🎯 Live Prediction":
            self.live_prediction()
        elif page == "📋 Data Explorer":
            self.data_explorer()


if __name__ == "__main__":
    dashboard = FraudDetectionDashboard()
    dashboard.run()