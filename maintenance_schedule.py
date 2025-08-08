import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_maintenance_schedule, calculate_cost_savings
from visualization import VisualizationManager

st.set_page_config(page_title="Maintenance Schedule", page_icon="📅", layout="wide")

def main():
    st.title("📅 Intelligent Maintenance Scheduling")
    st.markdown("Generate optimized maintenance schedules based on AI predictions.")
    
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
    
    if not st.session_state.model.is_trained:
        st.warning("⚠️ Model not trained. Please train a model first.")
        if st.button("Go to Model Training"):
            st.switch_page("pages/2_Model_Training.py")
        return
    
    # Scheduling configuration
    st.header("⚙️ Schedule Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Maintenance Thresholds")
        
        warning_threshold = st.slider("Warning Threshold", 0.4, 0.8, 0.6, 0.05)
        critical_threshold = st.slider("Critical Threshold", 0.6, 0.95, 0.8, 0.05)
        
        st.markdown(f"""
        **Threshold Explanation:**
        - **Warning ({warning_threshold:.0%})**: Schedule maintenance soon
        - **Critical ({critical_threshold:.0%})**: Immediate maintenance required
        """)
    
    with col2:
        st.subheader("Planning Horizon")
        
        planning_days = st.slider("Planning Period (days)", 7, 60, 14, 7)
        
        # Maintenance preferences
        maintenance_types = st.multiselect(
            "Maintenance Types",
            ['replacement', 'inspection', 'adjustment', 'cleaning'],
            default=['replacement', 'inspection']
        )
        
        # Working hours
        work_hours_per_day = st.slider("Available Work Hours/Day", 4, 12, 8, 1)
    
    # Generate schedule
    if st.button("📋 Generate Maintenance Schedule", type="primary"):
        generate_schedule(warning_threshold, critical_threshold, planning_days, maintenance_types, work_hours_per_day)
    
    # Display existing schedule
    if hasattr(st.session_state, 'maintenance_schedule') and not st.session_state.maintenance_schedule.empty:
        st.markdown("---")
        display_maintenance_schedule()
        
        st.markdown("---")
        display_cost_analysis()
        
        st.markdown("---")
        display_schedule_optimization()

def generate_schedule(warning_threshold, critical_threshold, planning_days, maintenance_types, work_hours_per_day):
    """Generate maintenance schedule based on predictions"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Analyzing tool conditions...")
        progress_bar.progress(25)
        
        # Generate maintenance schedule using warning threshold
        schedule = generate_maintenance_schedule(
            st.session_state.data, 
            st.session_state.model, 
            threshold=warning_threshold,
            days_ahead=planning_days
        )
        
        progress_bar.progress(50)
        status_text.text("📊 Optimizing schedule...")
        
        if len(schedule) > 0:
            # Add additional details to schedule
            schedule = enhance_schedule(schedule, critical_threshold, maintenance_types, work_hours_per_day)
            
            progress_bar.progress(75)
            status_text.text("✅ Schedule generated!")
            
            st.session_state.maintenance_schedule = schedule
            
            progress_bar.progress(100)
            st.success(f"🎉 Generated maintenance schedule for {len(schedule)} tools!")
            
        else:
            st.info("✅ No maintenance required within the specified period!")
            st.session_state.maintenance_schedule = pd.DataFrame()
        
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Schedule generation failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def enhance_schedule(schedule, critical_threshold, maintenance_types, work_hours_per_day):
    """Enhance schedule with additional details"""
    
    enhanced_schedule = schedule.copy()
    
    # Add maintenance type based on wear level
    enhanced_schedule['Maintenance Type'] = enhanced_schedule.apply(
        lambda row: assign_maintenance_type(row['Current Wear'], critical_threshold, maintenance_types), 
        axis=1
    )
    
    # Add technician assignment (simplified)
    technicians = ['Tech_01', 'Tech_02', 'Tech_03', 'Tech_04', 'Tech_05']
    enhanced_schedule['Assigned Technician'] = np.random.choice(technicians, len(enhanced_schedule))
    
    # Add part requirements
    enhanced_schedule['Required Parts'] = enhanced_schedule.apply(
        lambda row: get_required_parts(row['Tool Id'], row['Maintenance Type']), 
        axis=1
    )
    
    # Optimize scheduling within work hours
    enhanced_schedule = optimize_work_schedule(enhanced_schedule, work_hours_per_day)
    
    return enhanced_schedule

def assign_maintenance_type(wear_level_str, critical_threshold, maintenance_types):
    """Assign maintenance type based on wear level"""
    
    # Extract numeric wear level
    wear_level = float(wear_level_str.strip('%')) / 100
    
    if wear_level >= critical_threshold:
        return 'replacement'
    elif wear_level >= 0.6:
        if 'inspection' in maintenance_types:
            return 'inspection'
        else:
            return maintenance_types[0] if maintenance_types else 'replacement'
    else:
        if 'adjustment' in maintenance_types:
            return 'adjustment'
        else:
            return maintenance_types[-1] if maintenance_types else 'inspection'

def get_required_parts(tool_id, maintenance_type):
    """Get required parts for maintenance"""
    
    parts_map = {
        'replacement': ['New Tool Insert', 'Mounting Hardware', 'Coolant'],
        'inspection': ['Calibration Tools', 'Measurement Instruments'],
        'adjustment': ['Adjustment Tools', 'Lubricant'],
        'cleaning': ['Cleaning Solvents', 'Brushes', 'Protective Equipment']
    }
    
    base_parts = parts_map.get(maintenance_type, ['Standard Parts'])
    return ', '.join(base_parts)

def optimize_work_schedule(schedule, work_hours_per_day):
    """Optimize schedule within available work hours"""
    
    schedule = schedule.copy()
    
    # Convert scheduled date to datetime
    schedule['Scheduled Date'] = pd.to_datetime(schedule['Scheduled Date'])
    
    # Group by date and optimize
    daily_workload = {}
    
    for idx, row in schedule.iterrows():
        date = row['Scheduled Date'].date()
        duration = row['Estimated Duration']
        
        # Check if date has capacity
        current_load = daily_workload.get(date, 0)
        
        if current_load + duration > work_hours_per_day:
            # Move to next available day
            days_to_add = 1
            while True:
                new_date = date + timedelta(days=days_to_add)
                if daily_workload.get(new_date, 0) + duration <= work_hours_per_day:
                    schedule.loc[idx, 'Scheduled Date'] = pd.Timestamp(new_date)
                    daily_workload[new_date] = daily_workload.get(new_date, 0) + duration
                    break
                days_to_add += 1
        else:
            daily_workload[date] = current_load + duration
    
    # Format date back to string
    schedule['Scheduled Date'] = schedule['Scheduled Date'].dt.strftime('%Y-%m-%d')
    
    return schedule

def display_maintenance_schedule():
    """Display the generated maintenance schedule"""
    st.header("📋 Maintenance Schedule")
    
    schedule = st.session_state.maintenance_schedule
    
    if schedule.empty:
        st.info("No maintenance currently scheduled.")
        return
    
    # Schedule overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tasks", len(schedule))
    
    with col2:
        critical_count = len(schedule[schedule['Priority'] == 'Critical'])
        st.metric("Critical Tasks", critical_count)
    
    with col3:
        total_hours = schedule['Estimated Duration'].sum()
        st.metric("Total Hours", f"{total_hours}h")
    
    with col4:
        total_cost = schedule['Estimated Cost'].sum()
        st.metric("Total Cost", f"${total_cost:,}")
    
    # Priority breakdown
    st.subheader("📊 Priority Breakdown")
    
    priority_counts = schedule['Priority'].value_counts()
    
    fig_priority = px.pie(
        values=priority_counts.values,
        names=priority_counts.index,
        title="Tasks by Priority",
        color_discrete_map={
            'Critical': '#d62728',
            'High': '#ff7f0e',
            'Normal': '#2ca02c'
        }
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_priority, use_container_width=True)
    
    with col2:
        # Timeline visualization
        if len(schedule) > 0:
            fig_timeline = st.session_state.viz_manager.create_maintenance_timeline(schedule)
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Detailed schedule table
    st.subheader("📅 Detailed Schedule")
    
    # Add sorting and filtering
    col1, col2 = st.columns(2)
    
    with col1:
        sort_by = st.selectbox("Sort by", ['Priority', 'Scheduled Date', 'Estimated Cost', 'Current Wear'])
    
    with col2:
        filter_priority = st.multiselect("Filter by Priority", schedule['Priority'].unique(), default=schedule['Priority'].unique())
    
    # Apply filters
    filtered_schedule = schedule[schedule['Priority'].isin(filter_priority)]
    
    # Apply sorting
    if sort_by == 'Priority':
        priority_order = {'Critical': 0, 'High': 1, 'Normal': 2}
        filtered_schedule['sort_key'] = filtered_schedule['Priority'].map(priority_order)
        filtered_schedule = filtered_schedule.sort_values('sort_key').drop('sort_key', axis=1)
    else:
        filtered_schedule = filtered_schedule.sort_values(sort_by)
    
    # Display table
    st.dataframe(filtered_schedule, use_container_width=True)
    
    # Export options
    st.subheader("📤 Export Schedule")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = filtered_schedule.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"maintenance_schedule_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
    
    with col2:
        # Create summary report
        summary_report = create_schedule_summary(filtered_schedule)
        st.download_button(
            label="📊 Download Summary",
            data=summary_report,
            file_name=f"maintenance_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime='text/plain'
        )
    
    with col3:
        # Work order format
        work_orders = create_work_orders(filtered_schedule)
        st.download_button(
            label="📋 Download Work Orders",
            data=work_orders,
            file_name=f"work_orders_{datetime.now().strftime('%Y%m%d')}.txt",
            mime='text/plain'
        )

def display_cost_analysis():
    """Display cost analysis and savings"""
    st.header("💰 Cost Analysis")
    
    schedule = st.session_state.maintenance_schedule
    
    if schedule.empty:
        st.info("No cost analysis available - no maintenance scheduled.")
        return
    
    # Calculate cost savings
    cost_analysis = calculate_cost_savings(schedule)
    
    # Display savings metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Preventive Cost",
            f"${cost_analysis['preventive_cost']:,.0f}",
            help="Cost of planned preventive maintenance"
        )
    
    with col2:
        st.metric(
            "Reactive Cost",
            f"${cost_analysis['reactive_cost']:,.0f}",
            help="Estimated cost if maintenance was reactive"
        )
    
    with col3:
        st.metric(
            "Cost Savings",
            f"${cost_analysis['savings']:,.0f}",
            help="Total savings from predictive maintenance"
        )
    
    with col4:
        st.metric(
            "Savings %",
            f"{cost_analysis['savings_percentage']:.1f}%",
            help="Percentage cost reduction"
        )
    
    # Cost breakdown visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost comparison
        comparison_data = {
            'Maintenance Type': ['Predictive (Planned)', 'Reactive (Unplanned)'],
            'Total Cost': [cost_analysis['preventive_cost'], cost_analysis['reactive_cost']]
        }
        
        fig_comparison = px.bar(
            comparison_data,
            x='Maintenance Type',
            y='Total Cost',
            title="Cost Comparison: Predictive vs Reactive",
            color='Maintenance Type',
            color_discrete_map={
                'Predictive (Planned)': '#2ca02c',
                'Reactive (Unplanned)': '#d62728'
            }
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        # Cost by priority
        priority_costs = schedule.groupby('Priority')['Estimated Cost'].sum().reset_index()
        
        fig_priority_cost = px.pie(
            priority_costs,
            values='Estimated Cost',
            names='Priority',
            title="Cost Distribution by Priority",
            color_discrete_map={
                'Critical': '#d62728',
                'High': '#ff7f0e',
                'Normal': '#2ca02c'
            }
        )
        
        st.plotly_chart(fig_priority_cost, use_container_width=True)
    
    # ROI Analysis
    st.subheader("📈 Return on Investment (ROI)")
    
    # Estimated annual savings
    annual_savings = cost_analysis['savings'] * (365 / 14)  # Extrapolate from 14-day period
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Estimated Annual Savings", f"${annual_savings:,.0f}")
        
        # Implementation cost (estimated)
        implementation_cost = 50000  # Estimated system implementation cost
        roi_period = implementation_cost / annual_savings
        st.metric("ROI Payback Period", f"{roi_period:.1f} years")
    
    with col2:
        # Additional benefits
        st.markdown("**Additional Benefits:**")
        st.write("• 🎯 Improved product quality")
        st.write("• ⚡ Reduced downtime")
        st.write("• 🔧 Extended tool life")
        st.write("• 📊 Better planning visibility")
        st.write("• 👥 Optimized labor utilization")

def display_schedule_optimization():
    """Display schedule optimization options"""
    st.header("🔧 Schedule Optimization")
    
    schedule = st.session_state.maintenance_schedule
    
    if schedule.empty:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Workload Distribution")
        
        # Calculate daily workload
        schedule_copy = schedule.copy()
        schedule_copy['Scheduled Date'] = pd.to_datetime(schedule_copy['Scheduled Date'])
        
        daily_workload = schedule_copy.groupby(schedule_copy['Scheduled Date'].dt.date)['Estimated Duration'].sum()
        
        fig_workload = px.bar(
            x=daily_workload.index,
            y=daily_workload.values,
            title="Daily Workload Distribution",
            labels={'x': 'Date', 'y': 'Hours'}
        )
        
        fig_workload.add_hline(y=8, line_dash="dash", line_color="red", annotation_text="Max Capacity")
        
        st.plotly_chart(fig_workload, use_container_width=True)
    
    with col2:
        st.subheader("Optimization Options")
        
        # Resource constraints
        st.markdown("**Resource Constraints:**")
        max_daily_hours = st.slider("Max Daily Hours", 4, 12, 8)
        available_technicians = st.slider("Available Technicians", 1, 10, 3)
        
        # Optimization priorities
        st.markdown("**Optimization Priorities:**")
        optimization_mode = st.radio(
            "Optimize for:",
            ["Minimize Cost", "Minimize Downtime", "Balance Workload"],
            index=2
        )
        
        if st.button("🔄 Re-optimize Schedule"):
            st.info("Schedule optimization would be applied here with the selected constraints and priorities.")
    
    # Schedule insights
    st.subheader("💡 Schedule Insights")
    
    insights = generate_schedule_insights(schedule)
    
    for insight in insights:
        st.write(f"• {insight}")

def create_schedule_summary(schedule):
    """Create a text summary of the maintenance schedule"""
    
    summary = []
    summary.append("MAINTENANCE SCHEDULE SUMMARY")
    summary.append("=" * 40)
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    # Overview
    summary.append(f"Total Tasks: {len(schedule)}")
    summary.append(f"Critical Tasks: {len(schedule[schedule['Priority'] == 'Critical'])}")
    summary.append(f"High Priority Tasks: {len(schedule[schedule['Priority'] == 'High'])}")
    summary.append(f"Normal Priority Tasks: {len(schedule[schedule['Priority'] == 'Normal'])}")
    summary.append("")
    
    # Resource requirements
    summary.append(f"Total Duration: {schedule['Estimated Duration'].sum()} hours")
    summary.append(f"Total Cost: ${schedule['Estimated Cost'].sum():,}")
    summary.append("")
    
    # Task breakdown
    summary.append("SCHEDULED TASKS:")
    summary.append("-" * 20)
    
    for _, task in schedule.iterrows():
        summary.append(f"Tool {task['Tool Id']} - {task['Priority']} Priority")
        summary.append(f"  Date: {task['Scheduled Date']}")
        summary.append(f"  Type: {task['Maintenance Type']}")
        summary.append(f"  Duration: {task['Estimated Duration']} hours")
        summary.append(f"  Cost: ${task['Estimated Cost']}")
        summary.append("")
    
    return "\n".join(summary)

def create_work_orders(schedule):
    """Create work order format"""
    
    work_orders = []
    
    for i, task in schedule.iterrows():
        work_orders.append(f"WORK ORDER #{i+1:03d}")
        work_orders.append("=" * 30)
        work_orders.append(f"Tool ID: {task['Tool Id']}")
        work_orders.append(f"Machine: {task.get('Machine Id', 'TBD')}")
        work_orders.append(f"Scheduled Date: {task['Scheduled Date']}")
        work_orders.append(f"Priority: {task['Priority']}")
        work_orders.append(f"Maintenance Type: {task['Maintenance Type']}")
        work_orders.append(f"Estimated Duration: {task['Estimated Duration']} hours")
        work_orders.append(f"Estimated Cost: ${task['Estimated Cost']}")
        
        if 'Assigned Technician' in task:
            work_orders.append(f"Assigned Technician: {task['Assigned Technician']}")
        
        if 'Required Parts' in task:
            work_orders.append(f"Required Parts: {task['Required Parts']}")
        
        work_orders.append("")
        work_orders.append("Tasks to Complete:")
        work_orders.append("□ Pre-work safety check")
        work_orders.append("□ Remove worn tool")
        work_orders.append("□ Inspect tool holder")
        work_orders.append("□ Install new tool")
        work_orders.append("□ Calibrate and test")
        work_orders.append("□ Document completion")
        work_orders.append("")
        work_orders.append("Technician Signature: _______________")
        work_orders.append("Completion Date: _______________")
        work_orders.append("\n" + "="*50 + "\n")
    
    return "\n".join(work_orders)

def generate_schedule_insights(schedule):
    """Generate insights about the maintenance schedule"""
    
    insights = []
    
    # Peak workload analysis
    schedule_copy = schedule.copy()
    schedule_copy['Scheduled Date'] = pd.to_datetime(schedule_copy['Scheduled Date'])
    daily_workload = schedule_copy.groupby(schedule_copy['Scheduled Date'].dt.date)['Estimated Duration'].sum()
    
    if daily_workload.max() > 8:
        peak_date = daily_workload.idxmax()
        insights.append(f"⚠️ Peak workload of {daily_workload.max():.1f} hours on {peak_date}")
    
    # Cost concentration
    total_cost = schedule['Estimated Cost'].sum()
    critical_cost = schedule[schedule['Priority'] == 'Critical']['Estimated Cost'].sum()
    
    if critical_cost / total_cost > 0.5:
        insights.append(f"💰 {critical_cost/total_cost:.0%} of maintenance cost is for critical tasks")
    
    # Timeline analysis
    earliest_date = schedule_copy['Scheduled Date'].min().date()
    latest_date = schedule_copy['Scheduled Date'].max().date()
    schedule_span = (latest_date - earliest_date).days
    
    insights.append(f"📅 Maintenance scheduled over {schedule_span} days")
    
    # Resource utilization
    total_hours = schedule['Estimated Duration'].sum()
    avg_daily_hours = total_hours / max(1, schedule_span)
    
    if avg_daily_hours < 4:
        insights.append(f"⚡ Low resource utilization ({avg_daily_hours:.1f} hours/day average)")
    elif avg_daily_hours > 8:
        insights.append(f"🔥 High resource utilization ({avg_daily_hours:.1f} hours/day average)")
    
    return insights

if __name__ == "__main__":
    main()
