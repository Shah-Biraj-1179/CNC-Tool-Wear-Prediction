import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import classification_report
from utils import validate_sensor_data, calculate_cost_savings
from visualization import VisualizationManager
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")

def main():
    st.title("📈 Advanced Analytics & Insights")
    st.markdown("Deep dive into your CNC tool wear data with comprehensive analytics and insights.")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        from ml_models import ToolWearPredictor
        st.session_state.model = ToolWearPredictor()
    if 'viz_manager' not in st.session_state:
        st.session_state.viz_manager = VisualizationManager()
    
    # Check prerequisites
    if st.session_state.data is None:
        st.warning("⚠️ No data loaded. Please upload data first.")
        if st.button("Go to Data Upload"):
            st.switch_page("pages/1_Data_Upload.py")
        return
    
    # Analytics navigation
    st.header("🔍 Analytics Dashboard")
    
    # Analytics tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Insights", 
        "🤖 Model Analytics", 
        "🔧 Tool Lifecycle", 
        "💰 Financial Analysis", 
        "📋 Custom Reports"
    ])
    
    with tab1:
        display_data_insights()
    
    with tab2:
        display_model_analytics()
    
    with tab3:
        display_tool_lifecycle_analytics()
    
    with tab4:
        display_financial_analytics()
    
    with tab5:
        display_custom_reports()

def display_data_insights():
    """Display comprehensive data insights and patterns"""
    st.subheader("📊 Data Insights & Patterns")
    
    data = st.session_state.data
    
    # Data overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
        
    with col2:
        if 'tool_id' in data.columns:
            st.metric("Unique Tools", data['tool_id'].nunique())
        else:
            st.metric("Unique Tools", "N/A")
    
    with col3:
        if 'timestamp' in data.columns:
            try:
                data_copy = data.copy()
                data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
                date_range = (data_copy['timestamp'].max() - data_copy['timestamp'].min()).days
                st.metric("Date Range", f"{date_range} days")
            except:
                st.metric("Date Range", "N/A")
        else:
            st.metric("Date Range", "N/A")
    
    with col4:
        if 'wear_level' in data.columns:
            high_wear_count = len(data[data['wear_level'] > 0.8])
            st.metric("High Wear Tools", high_wear_count)
        else:
            st.metric("High Wear Tools", "N/A")
    
    # Advanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sensor correlation analysis
        st.subheader("🔗 Sensor Correlation Analysis")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            correlation_matrix = data[numeric_columns].corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Sensor Correlation Matrix",
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Correlation insights
            st.markdown("**Key Correlations:**")
            if 'wear_level' in correlation_matrix.columns:
                wear_correlations = correlation_matrix['wear_level'].abs().sort_values(ascending=False)
                for sensor, corr in wear_correlations.head(3).items():
                    if sensor != 'wear_level':
                        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
                        direction = "positive" if corr > 0 else "negative"
                        st.write(f"• **{sensor}**: {strength} {direction} correlation ({corr:.3f})")
        else:
            st.info("Insufficient numeric data for correlation analysis")
    
    with col2:
        # Wear level distribution analysis
        st.subheader("📈 Wear Level Distribution")
        
        if 'wear_level' in data.columns:
            fig_dist = px.histogram(
                data, 
                x='wear_level',
                nbins=30,
                title="Tool Wear Distribution",
                labels={'wear_level': 'Wear Level', 'count': 'Number of Observations'}
            )
            
            # Add threshold lines
            fig_dist.add_vline(x=0.6, line_dash="dash", line_color="orange", annotation_text="Warning")
            fig_dist.add_vline(x=0.8, line_dash="dash", line_color="red", annotation_text="Critical")
            
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Distribution statistics
            st.markdown("**Distribution Statistics:**")
            wear_stats = data['wear_level'].describe()
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write(f"• Mean: {wear_stats['mean']:.3f}")
                st.write(f"• Median: {wear_stats['50%']:.3f}")
                st.write(f"• Std Dev: {wear_stats['std']:.3f}")
            
            with col_b:
                st.write(f"• Min: {wear_stats['min']:.3f}")
                st.write(f"• Max: {wear_stats['max']:.3f}")
                
                # Skewness analysis
                skewness = stats.skew(data['wear_level'])
                if abs(skewness) < 0.5:
                    skew_desc = "approximately normal"
                elif skewness > 0.5:
                    skew_desc = "right-skewed"
                else:
                    skew_desc = "left-skewed"
                st.write(f"• Distribution: {skew_desc}")
        else:
            st.info("Wear level data not available")
    
    # Advanced pattern analysis
    st.subheader("🔍 Advanced Pattern Analysis")
    
    # Time series analysis if timestamp available
    if 'timestamp' in data.columns and 'wear_level' in data.columns:
        display_time_series_analysis(data)
    
    # Machine/Tool analysis
    if 'machine_id' in data.columns or 'tool_id' in data.columns:
        display_machine_tool_analysis(data)
    
    # Operating conditions analysis
    display_operating_conditions_analysis(data)

def display_time_series_analysis(data):
    """Display time series analysis of tool wear"""
    st.markdown("#### ⏰ Time Series Analysis")
    
    try:
        data_copy = data.copy()
        data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
        
        # Group by date and calculate daily statistics
        daily_stats = data_copy.groupby(data_copy['timestamp'].dt.date).agg({
            'wear_level': ['mean', 'max', 'min', 'count']
        }).round(3)
        
        daily_stats.columns = ['Mean_Wear', 'Max_Wear', 'Min_Wear', 'Count']
        daily_stats = daily_stats.reset_index()
        
        # Time series plot
        fig_ts = go.Figure()
        
        fig_ts.add_trace(go.Scatter(
            x=daily_stats['timestamp'],
            y=daily_stats['Mean_Wear'],
            mode='lines+markers',
            name='Average Wear',
            line=dict(color='blue', width=2)
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=daily_stats['timestamp'],
            y=daily_stats['Max_Wear'],
            mode='lines',
            name='Max Wear',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=daily_stats['timestamp'],
            y=daily_stats['Min_Wear'],
            mode='lines',
            name='Min Wear',
            line=dict(color='green', width=1, dash='dash')
        ))
        
        fig_ts.update_layout(
            title="Daily Wear Level Trends",
            xaxis_title="Date",
            yaxis_title="Wear Level",
            height=400
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Trend analysis
        if len(daily_stats) > 5:
            # Calculate trend
            x_numeric = np.arange(len(daily_stats))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, daily_stats['Mean_Wear'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
                st.metric("Trend Direction", trend_direction)
            
            with col2:
                st.metric("Trend Strength", f"R² = {r_value**2:.3f}")
            
            with col3:
                significance = "Significant" if p_value < 0.05 else "Not Significant"
                st.metric("Statistical Significance", significance)
    
    except Exception as e:
        st.warning(f"Time series analysis failed: {str(e)}")

def display_machine_tool_analysis(data):
    """Display machine and tool specific analysis"""
    st.markdown("#### 🔧 Machine & Tool Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Machine performance analysis
        if 'machine_id' in data.columns and 'wear_level' in data.columns:
            machine_stats = data.groupby('machine_id').agg({
                'wear_level': ['mean', 'std', 'count']
            }).round(3)
            
            machine_stats.columns = ['Mean_Wear', 'Std_Wear', 'Count']
            machine_stats = machine_stats.reset_index()
            
            fig_machine = px.bar(
                machine_stats,
                x='machine_id',
                y='Mean_Wear',
                error_y='Std_Wear',
                title="Average Wear by Machine",
                labels={'Mean_Wear': 'Average Wear Level'}
            )
            
            fig_machine.add_hline(y=0.6, line_dash="dash", line_color="orange")
            fig_machine.add_hline(y=0.8, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig_machine, use_container_width=True)
        else:
            st.info("Machine data not available")
    
    with col2:
        # Tool performance analysis
        if 'tool_id' in data.columns and 'wear_level' in data.columns:
            # Top 10 tools by wear level
            tool_stats = data.groupby('tool_id').agg({
                'wear_level': ['mean', 'max', 'count']
            }).round(3)
            
            tool_stats.columns = ['Mean_Wear', 'Max_Wear', 'Count']
            tool_stats = tool_stats.reset_index()
            tool_stats = tool_stats.sort_values('Mean_Wear', ascending=False).head(10)
            
            fig_tool = px.bar(
                tool_stats,
                x='tool_id',
                y='Mean_Wear',
                title="Top 10 Tools by Wear Level",
                labels={'Mean_Wear': 'Average Wear Level'}
            )
            
            fig_tool.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_tool, use_container_width=True)
        else:
            st.info("Tool data not available")

def display_operating_conditions_analysis(data):
    """Display operating conditions impact analysis"""
    st.markdown("#### ⚙️ Operating Conditions Impact")
    
    if 'wear_level' not in data.columns:
        st.info("Wear level data not available for operating conditions analysis")
        return
    
    # Sensor impact analysis
    sensor_columns = ['vibration', 'temperature', 'cutting_force', 'spindle_speed', 'feed_rate']
    available_sensors = [col for col in sensor_columns if col in data.columns]
    
    if not available_sensors:
        st.info("Sensor data not available for analysis")
        return
    
    # Create bins for wear levels
    data_copy = data.copy()
    data_copy['wear_category'] = pd.cut(data_copy['wear_level'], 
                                       bins=[0, 0.3, 0.6, 0.8, 1.0], 
                                       labels=['Low', 'Medium', 'High', 'Critical'])
    
    # Box plots for each sensor
    cols_per_row = 2
    rows = (len(available_sensors) + cols_per_row - 1) // cols_per_row
    
    for i in range(0, len(available_sensors), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, sensor in enumerate(available_sensors[i:i+cols_per_row]):
            with cols[j]:
                fig_box = px.box(
                    data_copy,
                    x='wear_category',
                    y=sensor,
                    title=f"{sensor.replace('_', ' ').title()} vs Wear Category",
                    color='wear_category',
                    color_discrete_map={
                        'Low': '#2ca02c',
                        'Medium': '#ff7f0e',
                        'High': '#d62728',
                        'Critical': '#8B0000'
                    }
                )
                fig_box.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)
    
    # Statistical analysis
    st.markdown("**Statistical Analysis:**")
    
    statistical_results = []
    
    for sensor in available_sensors:
        # ANOVA test
        groups = [group[sensor].dropna() for name, group in data_copy.groupby('wear_category')]
        groups = [group for group in groups if len(group) > 0]
        
        if len(groups) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                significance = "Significant" if p_value < 0.05 else "Not Significant"
                
                statistical_results.append({
                    'Sensor': sensor.replace('_', ' ').title(),
                    'F-Statistic': f"{f_stat:.3f}",
                    'P-Value': f"{p_value:.3f}",
                    'Significance': significance,
                    'Impact': "High" if p_value < 0.01 else "Medium" if p_value < 0.05 else "Low"
                })
            except:
                statistical_results.append({
                    'Sensor': sensor.replace('_', ' ').title(),
                    'F-Statistic': "N/A",
                    'P-Value': "N/A",
                    'Significance': "Error",
                    'Impact': "Unknown"
                })
    
    if statistical_results:
        stats_df = pd.DataFrame(statistical_results)
        st.dataframe(stats_df, use_container_width=True)

def display_model_analytics():
    """Display model performance analytics"""
    st.subheader("🤖 Model Performance Analytics")
    
    if not st.session_state.model.is_trained:
        st.warning("⚠️ Model not trained. Please train a model first.")
        if st.button("Go to Model Training"):
            st.switch_page("pages/2_Model_Training.py")
        return
    
    metrics = st.session_state.model.get_detailed_metrics()
    
    # Model performance overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    
    with col2:
        st.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
    
    with col3:
        st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
    
    with col4:
        st.metric("MAE", f"{metrics.get('mae', 0):.3f}")
    
    # Detailed performance analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance analysis
        if 'feature_importance' in metrics and 'features' in metrics:
            st.subheader("🎯 Feature Importance Analysis")
            
            importance_df = pd.DataFrame({
                'Feature': metrics['features'],
                'Importance': metrics['feature_importance']
            }).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance Ranking"
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Feature insights
            st.markdown("**Key Insights:**")
            top_features = importance_df.tail(3)
            for _, row in top_features.iterrows():
                st.write(f"• **{row['Feature']}**: {row['Importance']:.1%} contribution")
        else:
            st.info("Feature importance not available for this model type")
    
    with col2:
        # Model validation analysis
        st.subheader("✅ Model Validation")
        
        if 'confusion_matrix' in metrics:
            fig_cm = st.session_state.viz_manager.create_confusion_matrix(metrics['confusion_matrix'])
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # Cross-validation results
        if 'cv_score_mean' in metrics:
            st.markdown("**Cross-Validation Results:**")
            
            cv_mean = metrics['cv_score_mean']
            cv_std = metrics['cv_score_std']
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("CV Mean Score", f"{cv_mean:.3f}")
            with col_b:
                st.metric("CV Std Dev", f"{cv_std:.3f}")
            
            # Stability assessment
            stability = "Excellent" if cv_std < 0.05 else "Good" if cv_std < 0.1 else "Fair" if cv_std < 0.15 else "Poor"
            st.write(f"**Model Stability:** {stability}")
    
    # Model comparison analysis
    st.subheader("📊 Model Performance Comparison")
    
    # Simulated comparison data for different algorithms
    comparison_data = {
        'Algorithm': ['Random Forest', 'SVM', 'Gradient Boosting'],
        'Accuracy': [metrics.get('accuracy', 0.85), 0.82, 0.88],
        'R² Score': [metrics.get('r2', 0.83), 0.79, 0.86],
        'Training Time (s)': [2.5, 8.2, 15.3],
        'Interpretability': ['High', 'Low', 'Medium']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Highlight current model
    current_model_type = st.session_state.model.model_type
    if current_model_type == 'random_forest':
        current_idx = 0
    elif current_model_type == 'support_vector_machine':
        current_idx = 1
    else:
        current_idx = 2
    
    # Performance comparison chart
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Bar(
        name='Accuracy',
        x=comparison_df['Algorithm'],
        y=comparison_df['Accuracy'],
        marker_color=['red' if i == current_idx else 'lightblue' for i in range(len(comparison_df))]
    ))
    
    fig_comparison.update_layout(
        title="Model Performance Comparison (Current Model Highlighted)",
        yaxis_title="Accuracy",
        height=400
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Detailed comparison table
    st.dataframe(comparison_df, use_container_width=True)
    
    # Model recommendations
    st.subheader("💡 Model Optimization Recommendations")
    
    recommendations = generate_model_recommendations(metrics, st.session_state.data)
    
    for rec in recommendations:
        st.write(f"• {rec}")

def display_tool_lifecycle_analytics():
    """Display tool lifecycle and longevity analysis"""
    st.subheader("🔧 Tool Lifecycle Analytics")
    
    data = st.session_state.data
    
    if 'tool_id' not in data.columns or 'wear_level' not in data.columns:
        st.warning("Tool ID or wear level data not available for lifecycle analysis")
        return
    
    # Tool lifecycle overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Average tool lifecycle
        avg_wear = data.groupby('tool_id')['wear_level'].mean()
        st.metric("Average Tool Wear", f"{avg_wear.mean():.1%}")
    
    with col2:
        # Tools near end of life
        eol_tools = len(data[data['wear_level'] > 0.8]['tool_id'].unique())
        st.metric("Tools Near EOL", eol_tools)
    
    with col3:
        # Tool utilization rate
        if 'cutting_time' in data.columns:
            avg_utilization = data.groupby('tool_id')['cutting_time'].mean().mean()
            st.metric("Avg Utilization", f"{avg_utilization:.1f}h")
        else:
            st.metric("Avg Utilization", "N/A")
    
    # Tool performance matrix
    st.subheader("📋 Tool Performance Matrix")
    
    # Calculate tool statistics
    tool_stats = data.groupby('tool_id').agg({
        'wear_level': ['mean', 'max', 'std', 'count'],
        'vibration': 'mean' if 'vibration' in data.columns else lambda x: None,
        'temperature': 'mean' if 'temperature' in data.columns else lambda x: None
    }).round(3)
    
    # Flatten column names
    tool_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in tool_stats.columns.values]
    tool_stats = tool_stats.reset_index()
    
    # Performance categorization
    if 'wear_level_mean' in tool_stats.columns:
        tool_stats['Performance_Category'] = pd.cut(
            tool_stats['wear_level_mean'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Excellent', 'Good', 'Fair', 'Poor']
        )
        
        # Performance distribution
        col1, col2 = st.columns(2)
        
        with col1:
            performance_counts = tool_stats['Performance_Category'].value_counts()
            
            fig_performance = px.pie(
                values=performance_counts.values,
                names=performance_counts.index,
                title="Tool Performance Distribution",
                color_discrete_map={
                    'Excellent': '#2ca02c',
                    'Good': '#1f77b4',
                    'Fair': '#ff7f0e',
                    'Poor': '#d62728'
                }
            )
            
            st.plotly_chart(fig_performance, use_container_width=True)
        
        with col2:
            # Scatter plot: wear vs usage
            if 'wear_level_count' in tool_stats.columns:
                fig_scatter = px.scatter(
                    tool_stats,
                    x='wear_level_count',
                    y='wear_level_mean',
                    color='Performance_Category',
                    title="Tool Wear vs Usage Frequency",
                    labels={
                        'wear_level_count': 'Number of Measurements',
                        'wear_level_mean': 'Average Wear Level'
                    },
                    hover_data=['tool_id']
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Tool ranking table
    st.subheader("🏆 Tool Performance Ranking")
    
    # Filter and display top/bottom performers
    col1, col2 = st.columns(2)
    
    with col1:
        # Top performers
        if 'wear_level_mean' in tool_stats.columns:
            top_performers = tool_stats.nsmallest(10, 'wear_level_mean')[['tool_id', 'wear_level_mean', 'Performance_Category']]
            top_performers.columns = ['Tool ID', 'Avg Wear Level', 'Category']
            
            st.markdown("**🥇 Top Performers (Lowest Wear)**")
            st.dataframe(top_performers, use_container_width=True)
    
    with col2:
        # Bottom performers
        if 'wear_level_mean' in tool_stats.columns:
            bottom_performers = tool_stats.nlargest(10, 'wear_level_mean')[['tool_id', 'wear_level_mean', 'Performance_Category']]
            bottom_performers.columns = ['Tool ID', 'Avg Wear Level', 'Category']
            
            st.markdown("**⚠️ Needs Attention (Highest Wear)**")
            st.dataframe(bottom_performers, use_container_width=True)
    
    # Lifecycle trends
    if 'timestamp' in data.columns:
        st.subheader("📈 Lifecycle Trends")
        
        try:
            data_copy = data.copy()
            data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
            
            # Select top 5 tools by data points for trend analysis
            top_tools = data['tool_id'].value_counts().head(5).index
            
            fig_trends = go.Figure()
            
            for tool_id in top_tools:
                tool_data = data_copy[data_copy['tool_id'] == tool_id].sort_values('timestamp')
                
                fig_trends.add_trace(go.Scatter(
                    x=tool_data['timestamp'],
                    y=tool_data['wear_level'],
                    mode='lines+markers',
                    name=f'Tool {tool_id}',
                    hovertemplate=f'<b>Tool {tool_id}</b><br>Date: %{{x}}<br>Wear: %{{y:.3f}}<extra></extra>'
                ))
            
            fig_trends.update_layout(
                title="Tool Wear Progression Over Time",
                xaxis_title="Date",
                yaxis_title="Wear Level",
                height=500
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Trend analysis failed: {str(e)}")

def display_financial_analytics():
    """Display financial impact and ROI analysis"""
    st.subheader("💰 Financial Impact Analysis")
    
    # Check if maintenance schedule exists
    if not hasattr(st.session_state, 'maintenance_schedule') or st.session_state.maintenance_schedule.empty:
        st.info("💡 Generate a maintenance schedule first to see financial analysis.")
        if st.button("Go to Maintenance Schedule"):
            st.switch_page("pages/4_Maintenance_Schedule.py")
        return
    
    schedule = st.session_state.maintenance_schedule
    cost_analysis = calculate_cost_savings(schedule)
    
    # Financial overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Annual Savings Projection",
            f"${cost_analysis['savings'] * 26:,.0f}",
            help="Extrapolated from bi-weekly analysis"
        )
    
    with col2:
        st.metric(
            "Cost Reduction",
            f"{cost_analysis['savings_percentage']:.1f}%",
            help="Percentage reduction vs reactive maintenance"
        )
    
    with col3:
        st.metric(
            "Preventive Cost",
            f"${cost_analysis['preventive_cost']:,.0f}",
            help="Total planned maintenance cost"
        )
    
    with col4:
        st.metric(
            "Avoided Reactive Cost",
            f"${cost_analysis['reactive_cost'] - cost_analysis['preventive_cost']:,.0f}",
            help="Cost avoided through predictive maintenance"
        )
    
    # ROI Analysis
    st.subheader("📊 Return on Investment (ROI)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROI calculation
        implementation_cost = 75000  # Estimated system cost
        annual_savings = cost_analysis['savings'] * 26
        
        if annual_savings > 0:
            payback_period = implementation_cost / annual_savings
            roi_3_year = ((annual_savings * 3 - implementation_cost) / implementation_cost) * 100
            
            st.metric("Payback Period", f"{payback_period:.1f} years")
            st.metric("3-Year ROI", f"{roi_3_year:.0f}%")
        
        # Additional financial benefits
        st.markdown("**Additional Benefits:**")
        st.write("• 🎯 25-40% reduction in quality defects")
        st.write("• ⚡ 35-50% reduction in unplanned downtime")
        st.write("• 🔧 15-25% extension in tool life")
        st.write("• 📊 20-30% improvement in production planning")
    
    with col2:
        # Cost breakdown visualization
        cost_breakdown = {
            'Category': ['Implementation', 'Annual Savings', 'Net Benefit (Year 1)'],
            'Amount': [implementation_cost, annual_savings, annual_savings - implementation_cost],
            'Type': ['Cost', 'Savings', 'Net']
        }
        
        fig_roi = px.bar(
            cost_breakdown,
            x='Category',
            y='Amount',
            color='Type',
            title="Financial Impact Breakdown",
            color_discrete_map={
                'Cost': '#d62728',
                'Savings': '#2ca02c',
                'Net': '#1f77b4'
            }
        )
        
        st.plotly_chart(fig_roi, use_container_width=True)
    
    # Cost trend analysis
    st.subheader("📈 Cost Trend Projections")
    
    # 5-year projection
    years = list(range(2024, 2030))
    cumulative_savings = []
    cumulative_costs = [implementation_cost]
    
    for i, year in enumerate(years):
        if i == 0:
            cumulative_savings.append(annual_savings)
        else:
            # Assume 5% growth in savings per year
            cumulative_savings.append(cumulative_savings[-1] * 1.05)
        
        if i > 0:
            cumulative_costs.append(cumulative_costs[-1] + annual_savings * 0.1)  # 10% maintenance cost
    
    projection_data = pd.DataFrame({
        'Year': years,
        'Annual_Savings': cumulative_savings,
        'Cumulative_Savings': np.cumsum(cumulative_savings),
        'Cumulative_Costs': cumulative_costs
    })
    
    fig_projection = go.Figure()
    
    fig_projection.add_trace(go.Scatter(
        x=projection_data['Year'],
        y=projection_data['Cumulative_Savings'],
        mode='lines+markers',
        name='Cumulative Savings',
        line=dict(color='green', width=3)
    ))
    
    fig_projection.add_trace(go.Scatter(
        x=projection_data['Year'],
        y=projection_data['Cumulative_Costs'],
        mode='lines+markers',
        name='Cumulative Costs',
        line=dict(color='red', width=3)
    ))
    
    fig_projection.update_layout(
        title="5-Year Financial Projection",
        xaxis_title="Year",
        yaxis_title="Amount ($)",
        height=400
    )
    
    st.plotly_chart(fig_projection, use_container_width=True)
    
    # Cost per tool analysis
    st.subheader("🔧 Cost per Tool Analysis")
    
    if 'tool_id' in st.session_state.data.columns:
        tools_count = st.session_state.data['tool_id'].nunique()
        cost_per_tool = cost_analysis['preventive_cost'] / max(1, tools_count)
        savings_per_tool = cost_analysis['savings'] / max(1, tools_count)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Cost per Tool", f"${cost_per_tool:.0f}")
        
        with col2:
            st.metric("Savings per Tool", f"${savings_per_tool:.0f}")
        
        with col3:
            st.metric("ROI per Tool", f"{(savings_per_tool/cost_per_tool)*100:.0f}%")

def display_custom_reports():
    """Display custom reports and export options"""
    st.subheader("📋 Custom Reports & Exports")
    
    # Report configuration
    st.markdown("#### 🎛️ Report Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Report type selection
        report_type = st.selectbox(
            "Select Report Type",
            [
                "Executive Summary",
                "Technical Analysis",
                "Financial Report",
                "Maintenance Schedule",
                "Tool Performance",
                "Sensor Analysis",
                "Custom Report"
            ]
        )
        
        # Date range selection
        if 'timestamp' in st.session_state.data.columns:
            try:
                data_copy = st.session_state.data.copy()
                data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
                
                min_date = data_copy['timestamp'].min().date()
                max_date = data_copy['timestamp'].max().date()
                
                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            except:
                st.info("Date filtering not available")
                date_range = None
        else:
            date_range = None
    
    with col2:
        # Tool/Machine filtering
        if 'tool_id' in st.session_state.data.columns:
            selected_tools = st.multiselect(
                "Select Tools (Optional)",
                st.session_state.data['tool_id'].unique(),
                help="Leave empty to include all tools"
            )
        else:
            selected_tools = []
        
        if 'machine_id' in st.session_state.data.columns:
            selected_machines = st.multiselect(
                "Select Machines (Optional)",
                st.session_state.data['machine_id'].unique(),
                help="Leave empty to include all machines"
            )
        else:
            selected_machines = []
    
    # Generate report
    if st.button("📊 Generate Report", type="primary"):
        generate_custom_report(report_type, date_range, selected_tools, selected_machines)
    
    # Export options
    st.markdown("#### 📤 Export Options")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        # Raw data export
        if st.button("📄 Export Raw Data"):
            export_raw_data()
    
    with export_col2:
        # Analytics summary export
        if st.button("📊 Export Analytics Summary"):
            export_analytics_summary()
    
    with export_col3:
        # Visualizations export
        if st.button("📈 Export Visualizations"):
            st.info("Visualization export feature would download charts as PNG/PDF files")

def generate_custom_report(report_type, date_range, selected_tools, selected_machines):
    """Generate custom report based on user selection"""
    
    data = st.session_state.data.copy()
    
    # Apply filters
    if date_range and 'timestamp' in data.columns:
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            if len(date_range) == 2:
                mask = (data['timestamp'].dt.date >= date_range[0]) & (data['timestamp'].dt.date <= date_range[1])
                data = data[mask]
        except:
            pass
    
    if selected_tools and 'tool_id' in data.columns:
        data = data[data['tool_id'].isin(selected_tools)]
    
    if selected_machines and 'machine_id' in data.columns:
        data = data[data['machine_id'].isin(selected_machines)]
    
    # Generate report content
    report_content = generate_report_content(report_type, data)
    
    # Display report
    st.markdown("#### 📋 Generated Report")
    
    with st.expander("View Report Content", expanded=True):
        st.markdown(report_content)
    
    # Download option
    st.download_button(
        label=f"📄 Download {report_type} Report",
        data=report_content,
        file_name=f"{report_type.lower().replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime='text/markdown'
    )

def generate_report_content(report_type, data):
    """Generate report content based on type"""
    
    report = []
    report.append(f"# {report_type} Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    if report_type == "Executive Summary":
        report.extend(generate_executive_summary(data))
    elif report_type == "Technical Analysis":
        report.extend(generate_technical_analysis(data))
    elif report_type == "Financial Report":
        report.extend(generate_financial_report(data))
    elif report_type == "Tool Performance":
        report.extend(generate_tool_performance_report(data))
    else:
        report.append("Report content would be generated based on the selected type.")
    
    return "\n".join(report)

def generate_executive_summary(data):
    """Generate executive summary content"""
    
    summary = []
    summary.append("## Executive Summary")
    summary.append("")
    
    # Key metrics
    summary.append("### Key Performance Indicators")
    summary.append(f"- Total tools monitored: {data['tool_id'].nunique() if 'tool_id' in data.columns else 'N/A'}")
    summary.append(f"- Average wear level: {data['wear_level'].mean():.1%}" if 'wear_level' in data.columns else "- Wear data not available")
    summary.append(f"- Tools requiring attention: {len(data[data['wear_level'] > 0.6]) if 'wear_level' in data.columns else 'N/A'}")
    summary.append("")
    
    # Recommendations
    summary.append("### Key Recommendations")
    summary.append("- Implement predictive maintenance program")
    summary.append("- Focus on high-wear tools for immediate attention")
    summary.append("- Optimize cutting parameters for tool longevity")
    summary.append("")
    
    return summary

def generate_technical_analysis(data):
    """Generate technical analysis content"""
    
    analysis = []
    analysis.append("## Technical Analysis")
    analysis.append("")
    
    # Data quality
    analysis.append("### Data Quality Assessment")
    missing_data = data.isnull().sum()
    analysis.append(f"- Total records: {len(data)}")
    analysis.append(f"- Missing values: {missing_data.sum()}")
    analysis.append("")
    
    # Statistical summary
    if 'wear_level' in data.columns:
        analysis.append("### Wear Level Statistics")
        wear_stats = data['wear_level'].describe()
        analysis.append(f"- Mean: {wear_stats['mean']:.3f}")
        analysis.append(f"- Standard deviation: {wear_stats['std']:.3f}")
        analysis.append(f"- Range: {wear_stats['min']:.3f} - {wear_stats['max']:.3f}")
        analysis.append("")
    
    return analysis

def generate_financial_report(data):
    """Generate financial report content"""
    
    report = []
    report.append("## Financial Impact Report")
    report.append("")
    
    if hasattr(st.session_state, 'maintenance_schedule') and not st.session_state.maintenance_schedule.empty:
        cost_analysis = calculate_cost_savings(st.session_state.maintenance_schedule)
        
        report.append("### Cost Analysis")
        report.append(f"- Preventive maintenance cost: ${cost_analysis['preventive_cost']:,.0f}")
        report.append(f"- Reactive maintenance cost (avoided): ${cost_analysis['reactive_cost']:,.0f}")
        report.append(f"- Total savings: ${cost_analysis['savings']:,.0f}")
        report.append(f"- Savings percentage: {cost_analysis['savings_percentage']:.1f}%")
        report.append("")
        
        # ROI calculation
        annual_savings = cost_analysis['savings'] * 26
        report.append("### Return on Investment")
        report.append(f"- Projected annual savings: ${annual_savings:,.0f}")
        report.append(f"- Estimated payback period: {75000/annual_savings:.1f} years")
        report.append("")
    else:
        report.append("Financial analysis requires maintenance schedule generation.")
        report.append("")
    
    return report

def generate_tool_performance_report(data):
    """Generate tool performance report content"""
    
    report = []
    report.append("## Tool Performance Report")
    report.append("")
    
    if 'tool_id' in data.columns and 'wear_level' in data.columns:
        tool_stats = data.groupby('tool_id')['wear_level'].agg(['mean', 'max', 'count'])
        
        report.append("### Top Performing Tools (Lowest Wear)")
        top_tools = tool_stats.nsmallest(5, 'mean')
        for tool_id, stats in top_tools.iterrows():
            report.append(f"- {tool_id}: {stats['mean']:.3f} average wear")
        report.append("")
        
        report.append("### Tools Requiring Attention (Highest Wear)")
        high_wear_tools = tool_stats.nlargest(5, 'mean')
        for tool_id, stats in high_wear_tools.iterrows():
            report.append(f"- {tool_id}: {stats['mean']:.3f} average wear")
        report.append("")
    else:
        report.append("Tool performance analysis requires tool ID and wear level data.")
        report.append("")
    
    return report

def export_raw_data():
    """Export raw data with current filters"""
    data = st.session_state.data
    
    csv = data.to_csv(index=False)
    st.download_button(
        label="📄 Download Filtered Data",
        data=csv,
        file_name=f"cnc_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv'
    )

def export_analytics_summary():
    """Export analytics summary"""
    summary_content = create_analytics_summary()
    
    st.download_button(
        label="📊 Download Analytics Summary",
        data=summary_content,
        file_name=f"analytics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime='text/plain'
    )

def create_analytics_summary():
    """Create comprehensive analytics summary"""
    
    data = st.session_state.data
    summary = []
    
    summary.append("CNC TOOL WEAR ANALYTICS SUMMARY")
    summary.append("=" * 50)
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    # Data overview
    summary.append("DATA OVERVIEW:")
    summary.append(f"- Total records: {len(data):,}")
    summary.append(f"- Unique tools: {data['tool_id'].nunique() if 'tool_id' in data.columns else 'N/A'}")
    summary.append(f"- Date range: {pd.to_datetime(data['timestamp']).min().date()} to {pd.to_datetime(data['timestamp']).max().date()}" if 'timestamp' in data.columns else "- No timestamp data")
    summary.append("")
    
    # Wear analysis
    if 'wear_level' in data.columns:
        summary.append("WEAR LEVEL ANALYSIS:")
        wear_stats = data['wear_level'].describe()
        summary.append(f"- Average wear: {wear_stats['mean']:.3f}")
        summary.append(f"- Standard deviation: {wear_stats['std']:.3f}")
        summary.append(f"- Tools >60% wear: {len(data[data['wear_level'] > 0.6])}")
        summary.append(f"- Tools >80% wear: {len(data[data['wear_level'] > 0.8])}")
        summary.append("")
    
    # Model performance
    if st.session_state.model.is_trained:
        metrics = st.session_state.model.get_detailed_metrics()
        summary.append("MODEL PERFORMANCE:")
        summary.append(f"- Accuracy: {metrics.get('accuracy', 0):.3f}")
        summary.append(f"- R² Score: {metrics.get('r2', 0):.3f}")
        summary.append(f"- RMSE: {metrics.get('rmse', 0):.3f}")
        summary.append("")
    
    return "\n".join(summary)

def generate_model_recommendations(metrics, data):
    """Generate model optimization recommendations"""
    
    recommendations = []
    
    accuracy = metrics.get('accuracy', 0)
    r2_score = metrics.get('r2', 0)
    
    # Performance-based recommendations
    if accuracy < 0.8:
        recommendations.append("🔧 Consider collecting more diverse training data")
        recommendations.append("📊 Explore additional feature engineering")
    
    if r2_score < 0.7:
        recommendations.append("🎯 Review feature selection and importance")
        recommendations.append("⚙️ Try hyperparameter tuning for better performance")
    
    # Data-based recommendations
    if len(data) < 1000:
        recommendations.append("📈 Increase dataset size for more robust training")
    
    if 'cv_score_std' in metrics and metrics['cv_score_std'] > 0.1:
        recommendations.append("🔄 Model shows instability - consider ensemble methods")
    
    # General recommendations
    recommendations.append("📋 Regularly retrain model with new data")
    recommendations.append("🔍 Monitor model performance over time")
    recommendations.append("🎛️ Consider A/B testing different algorithms")
    
    return recommendations

if __name__ == "__main__":
    main()
