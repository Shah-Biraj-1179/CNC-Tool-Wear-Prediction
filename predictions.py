import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go
from ml_models import ToolWearPredictor
from sample_data import get_real_time_sample
from utils import determine_tool_status, estimate_remaining_life

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")

def main():
    st.title("🎯 Real-Time Predictions")
    st.markdown("Get instant tool wear predictions and monitoring insights.")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = ToolWearPredictor()
    if 'viz_manager' not in st.session_state:
        from visualization import VisualizationManager
        st.session_state.viz_manager = VisualizationManager()
    
    # Check prerequisites
    if st.session_state.data is None:
        st.warning("⚠️ No data loaded. Please upload data first.")
        if st.button("Go to Data Upload"):
            st.switch_page("pages/1_Data_Upload.py")
        return
    
    if not st.session_state.model.is_trained:
        st.warning("⚠️ Model not trained. Please train a model first.")
        if st.button("Go to Model Training"):
            st.switch_page("pages/2_Model_Training.py")
        return
    
    # Prediction modes
    st.header("🔧 Prediction Mode")
    
    prediction_mode = st.radio(
        "Select prediction method:",
        ["Interactive Parameter Adjustment", "Real-Time Simulation", "Batch Prediction"],
        horizontal=True
    )
    
    if prediction_mode == "Interactive Parameter Adjustment":
        interactive_prediction()
    elif prediction_mode == "Real-Time Simulation":
        real_time_simulation()
    else:
        batch_prediction()

def interactive_prediction():
    """Interactive parameter adjustment for predictions"""
    st.subheader("🎛️ Interactive Parameter Adjustment")
    st.markdown("Adjust machine parameters to see how they affect tool wear predictions.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Adjust Parameters:**")
        
        # Parameter controls
        vibration = st.slider("Vibration (mm/s)", 0.5, 10.0, 2.5, 0.1)
        temperature = st.slider("Temperature (°C)", 20, 100, 50, 1)
        cutting_force = st.slider("Cutting Force (N)", 100, 1000, 400, 10)
        spindle_speed = st.slider("Spindle Speed (RPM)", 500, 5000, 2500, 100)
        feed_rate = st.slider("Feed Rate (mm/min)", 50, 500, 200, 10)
        cutting_time = st.slider("Cutting Time (hours)", 1, 100, 25, 1)
        
        # Create prediction data
        prediction_data = pd.DataFrame([{
            'vibration': vibration,
            'temperature': temperature,
            'cutting_force': cutting_force,
            'spindle_speed': spindle_speed,
            'feed_rate': feed_rate,
            'cutting_time': cutting_time,
            'tool_id': 'INTERACTIVE',
            'machine_id': 'VIRTUAL'
        }])
        
        # Parameter impact indicators
        st.markdown("**Parameter Impact:**")
        
        # Calculate relative impact (simplified)
        base_params = [2.5, 50, 400, 2500, 200, 25]
        current_params = [vibration, temperature, cutting_force, spindle_speed, feed_rate, cutting_time]
        
        impact_colors = []
        for base, current in zip(base_params, current_params):
            ratio = current / base
            if ratio > 1.2:
                impact_colors.append("🔴")  # High impact
            elif ratio > 1.1:
                impact_colors.append("🟠")  # Medium impact
            elif ratio < 0.9:
                impact_colors.append("🟡")  # Low impact
            else:
                impact_colors.append("🟢")  # Normal
        
        param_names = ["Vibration", "Temperature", "Force", "Speed", "Feed", "Time"]
        for name, color in zip(param_names, impact_colors):
            st.write(f"{color} {name}")
    
    with col2:
        try:
            # Make prediction
            wear_prediction = st.session_state.model.predict(prediction_data)[0]
            
            # Display main prediction
            st.markdown("**Prediction Results:**")
            
            # Gauge chart
            fig_gauge = create_prediction_gauge(wear_prediction)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Status and recommendations
            status = determine_tool_status(wear_prediction)
            remaining_life = estimate_remaining_life(wear_prediction, cutting_time)
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Wear Level", f"{wear_prediction:.1%}")
                st.metric("Tool Status", status['status'])
            
            with col_b:
                st.metric("Remaining Life", f"{remaining_life:.1f} hours")
                st.metric("Alert Level", status['alert_level'])
            
            with col_c:
                confidence = get_prediction_confidence(prediction_data)
                st.metric("Confidence", f"{confidence:.1%}")
                st.metric("Action Required", status['action'])
            
            # Recommendations
            st.markdown("**Recommendations:**")
            recommendations = generate_recommendations(wear_prediction, prediction_data.iloc[0])
            for rec in recommendations:
                st.write(f"• {rec}")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

def real_time_simulation():
    """Real-time simulation with live updates"""
    st.subheader("📡 Real-Time Simulation")
    st.markdown("Simulate live tool monitoring with automatic updates.")
    
    # Control panel
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**Simulation Controls:**")
        
        if st.button("🔄 Update Data", type="primary"):
            # Generate new real-time sample
            real_time_data = get_real_time_sample()
            st.session_state.real_time_data = real_time_data
            st.rerun()
        
        auto_update = st.checkbox("Auto Update (every 5 seconds)")
        
        if auto_update:
            # Auto-refresh mechanism
            st.markdown("🔄 Auto-updating...")
            time.sleep(5)
            st.rerun()
        
        # Current sensor readings
        if hasattr(st.session_state, 'real_time_data'):
            st.markdown("**Current Readings:**")
            data = st.session_state.real_time_data
            st.write(f"🔊 Vibration: {data['vibration']:.1f} mm/s")
            st.write(f"🌡️ Temperature: {data['temperature']:.1f} °C")
            st.write(f"⚡ Force: {data['cutting_force']:.1f} N")
            st.write(f"🔄 Speed: {data['spindle_speed']} RPM")
            st.write(f"➡️ Feed: {data['feed_rate']} mm/min")
    
    with col2:
        if not hasattr(st.session_state, 'real_time_data'):
            st.session_state.real_time_data = get_real_time_sample()
        
        try:
            # Convert to DataFrame for prediction
            df = pd.DataFrame([st.session_state.real_time_data])
            
            # Make prediction
            wear_prediction = st.session_state.model.predict(df)[0]
            
            # Real-time dashboard
            display_real_time_dashboard(wear_prediction, st.session_state.real_time_data)
            
        except Exception as e:
            st.error(f"Real-time prediction failed: {str(e)}")

def batch_prediction():
    """Batch prediction on selected tools"""
    st.subheader("📊 Batch Prediction")
    st.markdown("Generate predictions for multiple tools at once.")
    
    # Tool selection
    available_tools = st.session_state.data['tool_id'].unique()
    selected_tools = st.multiselect(
        "Select tools for prediction:",
        available_tools,
        default=available_tools[:10] if len(available_tools) > 10 else available_tools
    )
    
    if not selected_tools:
        st.warning("Please select at least one tool.")
        return
    
    if st.button("🚀 Generate Batch Predictions", type="primary"):
        generate_batch_predictions(selected_tools)

def generate_batch_predictions(selected_tools):
    """Generate predictions for selected tools"""
    try:
        predictions = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, tool_id in enumerate(selected_tools):
            status_text.text(f"Processing tool {tool_id}...")
            
            # Get latest data for this tool
            tool_data = st.session_state.data[st.session_state.data['tool_id'] == tool_id].tail(1)
            
            if len(tool_data) > 0:
                # Make prediction
                wear_prediction = st.session_state.model.predict(tool_data)[0]
                status = determine_tool_status(wear_prediction)
                
                predictions.append({
                    'Tool ID': tool_id,
                    'Predicted Wear': f"{wear_prediction:.1%}",
                    'Status': status['status'],
                    'Alert Level': status['alert_level'],
                    'Action Required': status['action'],
                    'Machine ID': tool_data['machine_id'].iloc[0] if 'machine_id' in tool_data.columns else 'Unknown'
                })
            
            progress_bar.progress((i + 1) / len(selected_tools))
        
        # Display results
        progress_bar.empty()
        status_text.empty()
        
        if predictions:
            st.success(f"✅ Generated predictions for {len(predictions)} tools!")
            
            # Results table
            results_df = pd.DataFrame(predictions)
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                critical_count = len([p for p in predictions if p['Alert Level'] == 'HIGH'])
                st.metric("Critical Tools", critical_count)
            
            with col2:
                warning_count = len([p for p in predictions if p['Alert Level'] in ['MEDIUM', 'HIGH']])
                st.metric("Tools Needing Attention", warning_count)
            
            with col3:
                good_count = len([p for p in predictions if p['Alert Level'] == 'NONE'])
                st.metric("Tools in Good Condition", good_count)
            
            # Export option
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📄 Download Results",
                data=csv,
                file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
        
    except Exception as e:
        st.error(f"Batch prediction failed: {str(e)}")

def create_prediction_gauge(wear_level):
    """Create a gauge chart for wear level prediction"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = wear_level,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Tool Wear Level"},
        delta = {'reference': 0.6},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.6], 'color': "lightgreen"},
                {'range': [0.6, 0.8], 'color': "yellow"},
                {'range': [0.8, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def display_real_time_dashboard(wear_prediction, sensor_data):
    """Display real-time monitoring dashboard"""
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Wear level gauge
        fig_gauge = create_prediction_gauge(wear_prediction)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Status indicators
        status = determine_tool_status(wear_prediction)
        remaining_life = estimate_remaining_life(wear_prediction, sensor_data.get('cutting_time', 25))
        
        st.metric("Current Status", status['status'])
        st.metric("Remaining Life", f"{remaining_life:.1f} hours")
        st.metric("Alert Level", status['alert_level'])
        
        # Color-coded status
        status_colors = {
            'GOOD': '🟢',
            'FAIR': '🟡',
            'WORN': '🟠',
            'CRITICAL': '🔴'
        }
        
        st.markdown(f"### {status_colors.get(status['status'], '⚪')} {status['status']}")
        st.write(f"**Action:** {status['action']}")
    
    with col3:
        # Trend indicators (simulated)
        st.metric("Wear Rate", "0.12%/hour", delta="0.02%")
        st.metric("Efficiency", "87%", delta="-3%")
        st.metric("Runtime Today", "6.5 hours")
        
        # Next maintenance
        if wear_prediction > 0.6:
            days_to_maintenance = max(1, int((0.8 - wear_prediction) / 0.02))
            st.metric("Next Maintenance", f"{days_to_maintenance} days")
        else:
            st.metric("Next Maintenance", "Not scheduled")
    
    # Sensor readings timeline (simulated)
    st.subheader("📈 Sensor Trends")
    
    # Generate some trend data
    times = pd.date_range(end=datetime.now(), periods=20, freq='5min')
    
    fig = go.Figure()
    
    # Add sensor traces
    fig.add_trace(go.Scatter(
        x=times,
        y=np.random.normal(sensor_data['vibration'], 0.2, 20),
        mode='lines+markers',
        name='Vibration',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=times,
        y=np.random.normal(sensor_data['temperature'], 2, 20),
        mode='lines+markers',
        name='Temperature',
        yaxis='y2',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title="Recent Sensor Trends",
        xaxis_title="Time",
        yaxis=dict(title="Vibration (mm/s)", side="left"),
        yaxis2=dict(title="Temperature (°C)", side="right", overlaying="y"),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def get_prediction_confidence(data):
    """Calculate prediction confidence (simplified)"""
    # This is a simplified confidence calculation
    # In practice, you might use prediction intervals or ensemble variance
    
    # Check if values are within normal ranges
    ranges = {
        'vibration': (1.0, 5.0),
        'temperature': (30, 80),
        'cutting_force': (200, 800),
        'spindle_speed': (1000, 4000),
        'feed_rate': (100, 400)
    }
    
    confidence = 1.0
    
    for param, (min_val, max_val) in ranges.items():
        if param in data.iloc[0]:
            value = data.iloc[0][param]
            if value < min_val or value > max_val:
                confidence *= 0.8
    
    return max(0.5, confidence)

def generate_recommendations(wear_level, sensor_data):
    """Generate operational recommendations"""
    recommendations = []
    
    if wear_level > 0.8:
        recommendations.append("🔴 IMMEDIATE ACTION: Replace tool before next production run")
        recommendations.append("📋 Schedule emergency maintenance")
    elif wear_level > 0.6:
        recommendations.append("🟠 SCHEDULE: Plan tool replacement within 2-3 shifts")
        recommendations.append("📊 Increase monitoring frequency")
    elif wear_level > 0.4:
        recommendations.append("🟡 MONITOR: Watch for accelerated wear trends")
    else:
        recommendations.append("🟢 CONTINUE: Tool operating within normal parameters")
    
    # Parameter-specific recommendations
    if sensor_data['vibration'] > 4.0:
        recommendations.append("⚠️ HIGH VIBRATION: Check tool mounting and balance")
    
    if sensor_data['temperature'] > 70:
        recommendations.append("🌡️ HIGH TEMPERATURE: Increase coolant flow or reduce cutting speed")
    
    if sensor_data['cutting_force'] > 600:
        recommendations.append("⚡ HIGH FORCE: Consider reducing feed rate or using sharper tool")
    
    return recommendations

if __name__ == "__main__":
    main()
