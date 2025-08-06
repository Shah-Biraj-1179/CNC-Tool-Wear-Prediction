import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.ml_models import ToolWearPredictor
from utils.data_processing import DataProcessor
from utils.maintenance_scheduler import MaintenanceScheduler
from utils.visualization import VisualizationManager
from data.sample_data import get_sample_data

# Page configuration
st.set_page_config(
    page_title="CNC Tool Wear Prediction System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = ToolWearPredictor()
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'maintenance_scheduler' not in st.session_state:
    st.session_state.maintenance_scheduler = MaintenanceScheduler()
if 'viz_manager' not in st.session_state:
    st.session_state.viz_manager = VisualizationManager()
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'data' not in st.session_state:
    st.session_state.data = None

def main():
    st.title("🔧 AI-Powered CNC Tool Wear Prediction System")
    st.markdown("### Predictive Maintenance for Industrial CNC Operations")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard", "Data Upload", "Model Training", "Predictions", "Maintenance Schedule", "Model Insights"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Data Upload":
        show_data_upload()
    elif page == "Model Training":
        show_model_training()
    elif page == "Predictions":
        show_predictions()
    elif page == "Maintenance Schedule":
        show_maintenance_schedule()
    elif page == "Model Insights":
        show_model_insights()

def show_dashboard():
    st.header("📊 System Dashboard")
    
    if st.session_state.data is None:
        st.warning("Please upload data first to view dashboard metrics.")
        st.info("Go to 'Data Upload' page to get started.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Tools Monitored",
            value=len(st.session_state.data['tool_id'].unique()) if st.session_state.data is not None else 0
        )
    
    with col2:
        if st.session_state.trained:
            accuracy = st.session_state.model.get_model_accuracy()
            st.metric(
                label="Model Accuracy",
                value=f"{accuracy:.1%}" if accuracy else "N/A"
            )
        else:
            st.metric(label="Model Accuracy", value="Not Trained")
    
    with col3:
        if st.session_state.data is not None:
            high_wear_tools = len(st.session_state.data[st.session_state.data['wear_level'] > 0.8])
            st.metric(
                label="High Wear Tools",
                value=high_wear_tools,
                delta=f"{high_wear_tools} critical"
            )
    
    with col4:
        if st.session_state.data is not None:
            avg_wear = st.session_state.data['wear_level'].mean()
            st.metric(
                label="Average Wear Level",
                value=f"{avg_wear:.2f}",
                delta="Acceptable" if avg_wear < 0.7 else "High"
            )
    
    # Recent tool wear trends
    if st.session_state.data is not None:
        st.subheader("Tool Wear Trends")
        fig = st.session_state.viz_manager.create_wear_trend_chart(st.session_state.data)
        st.plotly_chart(fig, use_container_width=True)

def show_data_upload():
    st.header("📤 Data Upload & Management")
    
    # File upload option
    st.subheader("Upload CNC Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file with CNC machine data",
        type=['csv'],
        help="File should contain columns: tool_id, vibration, temperature, cutting_force, spindle_speed, feed_rate, wear_level, timestamp"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            # Validate data
            required_columns = ['tool_id', 'vibration', 'temperature', 'cutting_force', 'spindle_speed', 'feed_rate', 'wear_level']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.info("Required columns: tool_id, vibration, temperature, cutting_force, spindle_speed, feed_rate, wear_level, timestamp (optional)")
            else:
                # Process and validate data
                processed_data = st.session_state.data_processor.validate_and_clean_data(data)
                st.session_state.data = processed_data
                
                st.success(f"✅ Data uploaded successfully! {len(processed_data)} records loaded.")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(processed_data.head(10))
                
                # Show data statistics
                st.subheader("Data Statistics")
                st.dataframe(processed_data.describe())
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Option to use sample data
    st.subheader("Or Use Sample Data")
    if st.button("Load Sample CNC Data"):
        sample_data = get_sample_data()
        st.session_state.data = sample_data
        st.success("✅ Sample data loaded successfully!")
        st.info("Sample data includes 1000 records with various tool conditions.")
        
        # Show sample data preview
        st.dataframe(sample_data.head(10))

def show_model_training():
    st.header("🤖 Model Training & Evaluation")
    
    if st.session_state.data is None:
        st.warning("Please upload data first before training the model.")
        return
    
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type",
            ["Random Forest", "Support Vector Machine", "Gradient Boosting"]
        )
        
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        cross_validation = st.checkbox("Enable Cross-Validation", value=True)
        
        n_estimators = 100
        kernel = "rbf"
        learning_rate = 0.1
        
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
        elif model_type == "Support Vector Machine":
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        else:  # Gradient Boosting
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                # Prepare training parameters
                params = {'test_size': test_size, 'cv': cross_validation}
                
                if model_type == "Random Forest":
                    params['n_estimators'] = n_estimators
                elif model_type == "Support Vector Machine":
                    params['kernel'] = kernel
                else:
                    params['learning_rate'] = learning_rate
                
                # Train the model
                results = st.session_state.model.train_model(
                    st.session_state.data, 
                    model_type=model_type.lower().replace(' ', '_'),
                    **params
                )
                
                st.session_state.trained = True
                
                # Display results
                st.success("✅ Model trained successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.1%}")
                with col2:
                    st.metric("Precision", f"{results['precision']:.1%}")
                with col3:
                    st.metric("Recall", f"{results['recall']:.1%}")
                
                # Feature importance
                if 'feature_importance' in results:
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': results['features'],
                        'Importance': results['feature_importance']
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Importance', y='Feature', 
                               orientation='h', title="Feature Importance Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confusion matrix
                if 'confusion_matrix' in results:
                    st.subheader("Model Performance")
                    fig = st.session_state.viz_manager.create_confusion_matrix(results['confusion_matrix'])
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

def show_predictions():
    st.header("🔮 Tool Wear Predictions")
    
    if not st.session_state.trained:
        st.warning("Please train the model first before making predictions.")
        return
    
    st.subheader("Real-time Prediction Interface")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Machine Parameters**")
        vibration = st.slider("Vibration (mm/s)", 0.0, 10.0, 2.5, 0.1)
        temperature = st.slider("Temperature (°C)", 20.0, 100.0, 45.0, 1.0)
        cutting_force = st.slider("Cutting Force (N)", 100.0, 1000.0, 400.0, 10.0)
    
    with col2:
        st.write("**Operating Conditions**")
        spindle_speed = st.slider("Spindle Speed (RPM)", 500, 5000, 2000, 100)
        feed_rate = st.slider("Feed Rate (mm/min)", 50, 500, 200, 10)
        cutting_time = st.slider("Cutting Time (hours)", 0.0, 100.0, 10.0, 1.0)
    
    # Make prediction
    input_data = pd.DataFrame({
        'vibration': [vibration],
        'temperature': [temperature],
        'cutting_force': [cutting_force],
        'spindle_speed': [spindle_speed],
        'feed_rate': [feed_rate],
        'cutting_time': [cutting_time]
    })
    
    prediction = st.session_state.model.predict(input_data)
    confidence = st.session_state.model.predict_proba(input_data)
    
    # Display prediction results
    st.subheader("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        wear_level = prediction[0]
        st.metric("Predicted Wear Level", f"{wear_level:.2f}")
        
        if wear_level < 0.3:
            st.success("🟢 Good Condition")
        elif wear_level < 0.7:
            st.warning("🟡 Moderate Wear")
        else:
            st.error("🔴 High Wear - Maintenance Required")
    
    with col2:
        if confidence is not None:
            conf_score = np.max(confidence[0])
            st.metric("Prediction Confidence", f"{conf_score:.1%}")
    
    with col3:
        # Estimated remaining life
        if wear_level > 0:
            remaining_life = max(0, (1.0 - wear_level) / 0.1 * 10)  # Simplified calculation
            st.metric("Est. Remaining Life", f"{remaining_life:.1f} hours")
    
    # Prediction visualization
    st.subheader("Wear Level Visualization")
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = wear_level,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Tool Wear Level"},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_maintenance_schedule():
    st.header("📅 Maintenance Scheduling")
    
    if not st.session_state.trained:
        st.warning("Please train the model first before generating maintenance schedules.")
        return
    
    st.subheader("Generate Maintenance Schedule")
    
    col1, col2 = st.columns(2)
    
    with col1:
        schedule_days = st.slider("Schedule Period (days)", 7, 90, 30, 7)
        maintenance_threshold = st.slider("Maintenance Threshold", 0.5, 0.9, 0.8, 0.05)
    
    with col2:
        maintenance_buffer = st.slider("Maintenance Buffer (days)", 1, 10, 3, 1)
        priority_level = st.selectbox("Priority Level", ["All", "High", "Critical"])
    
    if st.button("Generate Schedule"):
        try:
            schedule = st.session_state.maintenance_scheduler.generate_schedule(
                st.session_state.data,
                st.session_state.model,
                days_ahead=schedule_days,
                threshold=maintenance_threshold,
                buffer_days=maintenance_buffer
            )
            
            if not schedule.empty:
                st.success(f"✅ Generated maintenance schedule for {len(schedule)} tools")
                
                # Display schedule table
                st.subheader("Maintenance Schedule")
                
                # Color code by priority
                def color_priority(val):
                    if val == "Critical":
                        return "background-color: #ffcccc"
                    elif val == "High":
                        return "background-color: #ffffcc"
                    else:
                        return "background-color: #ccffcc"
                
                styled_schedule = schedule.style.map(
                    color_priority, subset=['Priority']
                )
                
                st.dataframe(styled_schedule, use_container_width=True)
                
                # Schedule visualization
                st.subheader("Schedule Timeline")
                fig = st.session_state.viz_manager.create_maintenance_timeline(schedule)
                st.plotly_chart(fig, use_container_width=True)
                
                # Export functionality
                st.subheader("Export Schedule")
                csv_data = schedule.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name=f"maintenance_schedule_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Summary statistics
                st.subheader("Schedule Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    critical_count = len(schedule[schedule['Priority'] == 'Critical'])
                    st.metric("Critical Maintenance", critical_count)
                
                with col2:
                    high_count = len(schedule[schedule['Priority'] == 'High'])
                    st.metric("High Priority", high_count)
                
                with col3:
                    normal_count = len(schedule[schedule['Priority'] == 'Normal'])
                    st.metric("Normal Priority", normal_count)
                
            else:
                st.info("No maintenance required in the specified period.")
                
        except Exception as e:
            st.error(f"Error generating schedule: {str(e)}")

def show_model_insights():
    st.header("🧠 Model Insights & Explanations")
    
    if not st.session_state.trained:
        st.warning("Please train the model first to view insights.")
        return
    
    st.subheader("Model Performance Analysis")
    
    # Model metrics
    if hasattr(st.session_state.model, 'get_detailed_metrics'):
        metrics = st.session_state.model.get_detailed_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
        with col2:
            st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
        with col3:
            st.metric("MAE", f"{metrics.get('mae', 0):.3f}")
        with col4:
            st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
    
    # Feature analysis
    st.subheader("Feature Impact Analysis")
    
    if st.session_state.data is not None:
        # Correlation heatmap
        numeric_columns = st.session_state.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = st.session_state.data[numeric_columns].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model interpretation
    st.subheader("Prediction Factors")
    
    st.info("""
    **Key Factors Affecting Tool Wear:**
    
    🔹 **Vibration**: Higher vibration levels typically indicate increased tool wear
    🔹 **Temperature**: Excessive heat accelerates tool degradation
    🔹 **Cutting Force**: Higher forces put more stress on the cutting tool
    🔹 **Spindle Speed**: Optimal speeds reduce wear, while extreme speeds increase it
    🔹 **Feed Rate**: Balanced feed rates optimize tool life
    """)
    
    # Recommendations
    st.subheader("Optimization Recommendations")
    
    if st.session_state.data is not None:
        # Generate recommendations based on data patterns
        high_wear_data = st.session_state.data[st.session_state.data['wear_level'] > 0.7]
        
        if not high_wear_data.empty:
            avg_vibration = high_wear_data['vibration'].mean()
            avg_temp = high_wear_data['temperature'].mean()
            avg_force = high_wear_data['cutting_force'].mean()
            
            recommendations = []
            
            if avg_vibration > 5.0:
                recommendations.append("⚠️ Consider reducing machine vibration through better mounting or balancing")
            
            if avg_temp > 70:
                recommendations.append("🌡️ Implement better cooling systems to reduce operating temperature")
            
            if avg_force > 600:
                recommendations.append("⚡ Optimize cutting parameters to reduce cutting forces")
            
            if recommendations:
                st.warning("**Recommendations for High Wear Tools:**")
                for rec in recommendations:
                    st.write(rec)
            else:
                st.success("✅ Current operating conditions are within optimal ranges")

if __name__ == "__main__":
    main()
