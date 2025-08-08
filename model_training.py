import streamlit as st
import pandas as pd
import numpy as np
import time
from ml_models import ToolWearPredictor
from visualization import VisualizationManager

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

def main():
    st.title("🤖 Model Training")
    st.markdown("Train machine learning models to predict tool wear levels.")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = ToolWearPredictor()
    if 'viz_manager' not in st.session_state:
        st.session_state.viz_manager = VisualizationManager()
    
    # Check if data is available
    if st.session_state.data is None:
        st.warning("⚠️ No data loaded. Please upload data first.")
        if st.button("Go to Data Upload"):
            st.switch_page("pages/1_Data_Upload.py")
        return
    
    # Training configuration
    st.header("🔧 Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Selection")
        model_type = st.selectbox(
            "Choose Algorithm",
            ['random_forest', 'support_vector_machine', 'gradient_boosting'],
            format_func=lambda x: {
                'random_forest': 'Random Forest (Recommended)',
                'support_vector_machine': 'Support Vector Machine',
                'gradient_boosting': 'Gradient Boosting'
            }[x]
        )
        
        # Model-specific parameters
        if model_type == 'random_forest':
            n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
            params = {'n_estimators': n_estimators}
        elif model_type == 'support_vector_machine':
            kernel = st.selectbox("Kernel", ['rbf', 'poly', 'linear'])
            params = {'kernel': kernel}
        else:  # gradient_boosting
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
            params = {'learning_rate': learning_rate}
    
    with col2:
        st.subheader("Training Options")
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        enable_cv = st.checkbox("Enable Cross-Validation", value=True)
        
        # Data preprocessing info
        st.info("📋 **Automatic Preprocessing:**\n"
               "• Feature scaling (for SVM)\n"
               "• Missing value handling\n"
               "• Feature engineering\n"
               "• Data validation")
    
    # Train model
    if st.button("🚀 Train Model", type="primary"):
        train_model(model_type, params, test_size, enable_cv)
    
    # Display training results
    if st.session_state.model.is_trained:
        st.markdown("---")
        display_training_results()
        
        st.markdown("---")
        display_model_insights()

def train_model(model_type, params, test_size, enable_cv):
    """Train the selected model"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Update progress
        status_text.text("🔄 Preparing data...")
        progress_bar.progress(20)
        
        # Validate data
        required_columns = ['vibration', 'temperature', 'cutting_force', 'spindle_speed', 'feed_rate', 'wear_level']
        missing_columns = [col for col in required_columns if col not in st.session_state.data.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            return
        
        # Start training
        status_text.text("🤖 Training model...")
        progress_bar.progress(40)
        
        start_time = time.time()
        
        # Train model
        results = st.session_state.model.train_model(
            st.session_state.data,
            model_type=model_type,
            test_size=test_size,
            cv=enable_cv,
            **params
        )
        
        training_time = time.time() - start_time
        
        # Update progress
        status_text.text("✅ Training completed!")
        progress_bar.progress(100)
        
        # Success message
        st.success(f"🎉 Model trained successfully in {training_time:.2f} seconds!")
        
        # Display immediate results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Accuracy", f"{results['accuracy']:.1%}")
        
        with col2:
            st.metric("R² Score", f"{results['r2']:.3f}")
        
        with col3:
            st.metric("RMSE", f"{results['rmse']:.3f}")
        
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        status_text.empty()
        progress_bar.empty()

def display_training_results():
    """Display detailed training results"""
    st.header("📊 Training Results")
    
    metrics = st.session_state.model.get_detailed_metrics()
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Metrics")
        
        # Main metrics
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            st.metric("Precision", f"{metrics['precision']:.1%}")
        
        with metric_col2:
            st.metric("Recall", f"{metrics['recall']:.1%}")
            st.metric("R² Score", f"{metrics['r2']:.3f}")
        
        # Confusion Matrix
        if 'confusion_matrix' in metrics:
            st.subheader("Confusion Matrix")
            fig_cm = st.session_state.viz_manager.create_confusion_matrix(metrics['confusion_matrix'])
            st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.subheader("Regression Metrics")
        
        # Error metrics
        st.metric("Mean Absolute Error", f"{metrics['mae']:.3f}")
        st.metric("Root Mean Square Error", f"{metrics['rmse']:.3f}")
        
        # Cross-validation results
        if 'cv_score_mean' in metrics:
            st.subheader("Cross-Validation")
            st.metric("CV Score (Mean)", f"{metrics['cv_score_mean']:.3f}")
            st.metric("CV Score (Std)", f"{metrics['cv_score_std']:.3f}")
            
            st.info(f"📈 **Model Stability:** The cross-validation standard deviation of {metrics['cv_score_std']:.3f} indicates {'good' if metrics['cv_score_std'] < 0.1 else 'moderate'} model stability.")

def display_model_insights():
    """Display model insights and feature importance"""
    st.header("🔍 Model Insights")
    
    metrics = st.session_state.model.get_detailed_metrics()
    
    # Feature importance (for tree-based models)
    if 'feature_importance' in metrics and 'features' in metrics:
        st.subheader("Feature Importance")
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': metrics['features'],
            'Importance': metrics['feature_importance']
        }).sort_values('Importance', ascending=False)
        
        # Display as bar chart
        import plotly.express as px
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance Ranking",
            labels={'Importance': 'Importance Score', 'Feature': 'Features'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        st.subheader("Feature Importance Details")
        
        # Add interpretations
        importance_df['Interpretation'] = importance_df.apply(
            lambda row: interpret_feature_importance(row['Feature'], row['Importance']), axis=1
        )
        
        st.dataframe(importance_df, use_container_width=True)
    
    # Model recommendations
    st.subheader("💡 Model Recommendations")
    
    accuracy = metrics.get('accuracy', 0)
    r2_score = metrics.get('r2', 0)
    
    recommendations = []
    
    if accuracy > 0.9:
        recommendations.append("✅ Excellent model performance - ready for production use")
    elif accuracy > 0.8:
        recommendations.append("👍 Good model performance - suitable for most applications")
    else:
        recommendations.append("⚠️ Model performance could be improved - consider more data or feature engineering")
    
    if r2_score > 0.8:
        recommendations.append("✅ Strong predictive power for wear level estimation")
    elif r2_score > 0.6:
        recommendations.append("👍 Moderate predictive power - acceptable for trend analysis")
    else:
        recommendations.append("⚠️ Limited predictive power - consider additional features or different algorithm")
    
    # Data-specific recommendations
    data_size = len(st.session_state.data)
    if data_size < 500:
        recommendations.append("📊 Consider collecting more data to improve model reliability")
    
    for rec in recommendations:
        st.write(rec)
    
    # Model comparison
    st.subheader("🏁 Model Comparison")
    
    comparison_data = {
        'Algorithm': ['Random Forest', 'SVM', 'Gradient Boosting'],
        'Typical Accuracy': ['85-92%', '80-87%', '88-94%'],
        'Training Speed': ['Fast', 'Moderate', 'Slow'],
        'Interpretability': ['High', 'Low', 'Medium'],
        'Best For': ['General Purpose', 'Complex Patterns', 'High Accuracy']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

def interpret_feature_importance(feature, importance):
    """Provide interpretation for feature importance"""
    interpretations = {
        'vibration': f"Vibration directly indicates tool condition - {importance:.1%} contribution",
        'temperature': f"Heat generation affects tool life - {importance:.1%} contribution",
        'cutting_force': f"Mechanical stress indicator - {importance:.1%} contribution",
        'spindle_speed': f"Operating parameter influence - {importance:.1%} contribution",
        'feed_rate': f"Material advance rate impact - {importance:.1%} contribution",
        'cutting_time': f"Usage duration factor - {importance:.1%} contribution",
        'vibration_temp_ratio': f"Combined sensor interaction - {importance:.1%} contribution",
        'force_speed_ratio': f"Mechanical efficiency indicator - {importance:.1%} contribution",
        'power_estimate': f"Power consumption estimate - {importance:.1%} contribution"
    }
    
    return interpretations.get(feature, f"Feature contribution: {importance:.1%}")

if __name__ == "__main__":
    main()
