import streamlit as st
import pandas as pd
import io
from sample_data import get_sample_data
from utils import validate_sensor_data
from visualization import VisualizationManager
from ml_models import ToolWearPredictor

st.set_page_config(page_title="Data Upload", page_icon="📊", layout="wide")

def main():
    st.title("📊 Data Upload & Management")
    st.markdown("Upload your CNC sensor data or use sample data to get started.")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = ToolWearPredictor()
    if 'viz_manager' not in st.session_state:
        st.session_state.viz_manager = VisualizationManager()
    
    # Data upload options
    st.header("Data Source")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with CNC sensor data"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success(f"✅ File uploaded successfully! {len(data)} records loaded.")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(data.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with col2:
        st.subheader("Use Sample Data")
        st.markdown("Generate realistic CNC tool wear data for testing and demonstration.")
        
        # Sample data options
        n_records = st.slider("Number of Records", 100, 2000, 1000, 100)
        n_tools = st.slider("Number of Tools", 5, 50, 20, 5)
        
        if st.button("🎲 Generate Sample Data", type="primary"):
            with st.spinner("Generating sample data..."):
                sample_data = get_sample_data(n_records, n_tools)
                st.session_state.data = sample_data
                st.success(f"✅ Sample data generated! {len(sample_data)} records for {n_tools} tools.")
                st.rerun()
    
    # Data overview
    if st.session_state.data is not None:
        st.markdown("---")
        display_data_overview()
        
        # Data quality check
        st.markdown("---")
        display_data_quality()
        
        # Data export
        st.markdown("---")
        display_export_options()

def display_data_overview():
    """Display overview of loaded data"""
    st.header("📋 Data Overview")
    
    data = st.session_state.data
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(data))
    
    with col2:
        if 'tool_id' in data.columns:
            st.metric("Unique Tools", data['tool_id'].nunique())
        else:
            st.metric("Unique Tools", "N/A")
    
    with col3:
        if 'wear_level' in data.columns:
            avg_wear = data['wear_level'].mean()
            st.metric("Average Wear Level", f"{avg_wear:.1%}")
        else:
            st.metric("Average Wear Level", "N/A")
    
    with col4:
        if 'timestamp' in data.columns:
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                date_range = data['timestamp'].max() - data['timestamp'].min()
                st.metric("Date Range", f"{date_range.days} days")
            except:
                st.metric("Date Range", "N/A")
        else:
            st.metric("Date Range", "N/A")
    
    # Column information
    st.subheader("Column Information")
    col_info = []
    for col in data.columns:
        col_info.append({
            'Column': col,
            'Type': str(data[col].dtype),
            'Non-null Count': data[col].count(),
            'Null Count': data[col].isnull().sum(),
            'Unique Values': data[col].nunique()
        })
    
    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df, use_container_width=True)
    
    # Data sample
    st.subheader("Data Sample")
    st.dataframe(data.head(10), use_container_width=True)

def display_data_quality():
    """Display data quality assessment"""
    st.header("🔍 Data Quality Assessment")
    
    data = st.session_state.data
    
    # Validate data
    validation_results = validate_sensor_data(data)
    
    # Quality summary
    col1, col2 = st.columns(2)
    
    with col1:
        if validation_results['valid']:
            st.success("✅ Data quality check passed!")
        else:
            st.error("❌ Data quality issues detected")
        
        # Missing values
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            st.warning(f"⚠️ {missing_data.sum()} total missing values found")
            st.dataframe(missing_data[missing_data > 0].to_frame('Missing Count'))
        else:
            st.success("✅ No missing values")
    
    with col2:
        # Warnings and errors
        if validation_results['warnings']:
            st.warning("⚠️ Data Quality Warnings:")
            for warning in validation_results['warnings']:
                st.write(f"• {warning}")
        
        if validation_results['errors']:
            st.error("❌ Data Quality Errors:")
            for error in validation_results['errors']:
                st.write(f"• {error}")
    
    # Visualizations
    if 'wear_level' in data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Wear distribution
            fig_dist = st.session_state.viz_manager.create_wear_distribution(data)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Correlation heatmap
            fig_corr = st.session_state.viz_manager.create_correlation_heatmap(data)
            st.plotly_chart(fig_corr, use_container_width=True)

def display_export_options():
    """Display data export options"""
    st.header("💾 Export Data")
    
    data = st.session_state.data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Download Current Dataset")
        
        # CSV export
        csv = data.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name=f"cnc_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
        
        # JSON export
        json_str = data.to_json(orient='records', indent=2)
        st.download_button(
            label="📋 Download as JSON",
            data=json_str,
            file_name=f"cnc_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime='application/json'
        )
    
    with col2:
        st.subheader("Data Summary Report")
        
        # Generate summary report
        summary_report = generate_summary_report(data)
        
        st.download_button(
            label="📊 Download Summary Report",
            data=summary_report,
            file_name=f"data_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime='text/plain'
        )

def generate_summary_report(data):
    """Generate a text summary report of the data"""
    report = []
    report.append("CNC Tool Wear Data Summary Report")
    report.append("=" * 40)
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Basic info
    report.append("Dataset Overview:")
    report.append(f"- Total Records: {len(data)}")
    report.append(f"- Number of Columns: {len(data.columns)}")
    
    if 'tool_id' in data.columns:
        report.append(f"- Unique Tools: {data['tool_id'].nunique()}")
    
    if 'wear_level' in data.columns:
        report.append(f"- Average Wear Level: {data['wear_level'].mean():.3f}")
        report.append(f"- Max Wear Level: {data['wear_level'].max():.3f}")
        report.append(f"- High Wear Tools (>80%): {len(data[data['wear_level'] > 0.8])}")
    
    report.append("")
    
    # Missing data
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        report.append("Missing Data:")
        for col, missing in missing_data.items():
            if missing > 0:
                report.append(f"- {col}: {missing} ({missing/len(data)*100:.1f}%)")
    else:
        report.append("Missing Data: None")
    
    report.append("")
    
    # Column statistics
    report.append("Numeric Column Statistics:")
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        report.append(f"- {col}:")
        report.append(f"  Mean: {data[col].mean():.3f}")
        report.append(f"  Std: {data[col].std():.3f}")
        report.append(f"  Min: {data[col].min():.3f}")
        report.append(f"  Max: {data[col].max():.3f}")
        report.append("")
    
    return "\n".join(report)

if __name__ == "__main__":
    main()
