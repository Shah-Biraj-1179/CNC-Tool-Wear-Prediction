import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class VisualizationManager:
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.wear_colors = {
            'low': '#2ca02c',      # Green
            'medium': '#ff7f0e',   # Orange
            'high': '#d62728'      # Red
        }
    
    def create_wear_trend_chart(self, data):
        """Create tool wear trend visualization"""
        try:
            # Prepare data for visualization
            if 'timestamp' not in data.columns:
                # If no timestamp, create a sequence
                data = data.copy()
                data['timestamp'] = pd.date_range(start='2024-01-01', periods=len(data), freq='H')
            
            # Group by tool_id and create trend lines
            fig = go.Figure()
            
            # Get unique tools (limit to top 10 for readability)
            top_tools = data['tool_id'].value_counts().head(10).index
            
            for tool_id in top_tools:
                tool_data = data[data['tool_id'] == tool_id].sort_values('timestamp')
                
                # Color based on latest wear level
                latest_wear = tool_data['wear_level'].iloc[-1]
                if latest_wear < 0.6:
                    color = self.wear_colors['low']
                elif latest_wear < 0.8:
                    color = self.wear_colors['medium']
                else:
                    color = self.wear_colors['high']
                
                fig.add_trace(go.Scatter(
                    x=tool_data['timestamp'],
                    y=tool_data['wear_level'],
                    mode='lines+markers',
                    name=f'Tool {tool_id}',
                    line=dict(color=color, width=2),
                    hovertemplate='<b>Tool %{fullData.name}</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Wear Level: %{y:.3f}<br>' +
                                  '<extra></extra>'
                ))
            
            # Add warning zones
            fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                         annotation_text="Critical Threshold", annotation_position="bottom right")
            fig.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                         annotation_text="Warning Threshold", annotation_position="bottom right")
            
            fig.update_layout(
                title="Tool Wear Trends Over Time",
                xaxis_title="Time",
                yaxis_title="Wear Level",
                hovermode='x unified',
                showlegend=True,
                height=500
            )
            
            return fig
            
        except Exception as e:
            # Return empty chart on error
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating chart: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
    
    def create_sensor_dashboard(self, data):
        """Create multi-sensor dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Vibration', 'Temperature', 'Cutting Force', 'Spindle Speed'),
                vertical_spacing=0.1
            )
            
            sensors = [
                ('vibration', 'Vibration (mm/s)'),
                ('temperature', 'Temperature (°C)'),
                ('cutting_force', 'Cutting Force (N)'),
                ('spindle_speed', 'Spindle Speed (RPM)')
            ]
            
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for i, ((sensor, title), (row, col)) in enumerate(zip(sensors, positions)):
                if sensor in data.columns:
                    # Box plot for each sensor
                    fig.add_trace(
                        go.Box(
                            y=data[sensor],
                            name=title,
                            showlegend=False,
                            boxpoints='outliers'
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title="Sensor Readings Distribution",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating dashboard: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
    
    def create_maintenance_timeline(self, schedule_df):
        """Create maintenance schedule timeline"""
        try:
            if schedule_df.empty:
                fig = go.Figure()
                fig.add_annotation(text="No maintenance scheduled", 
                                 xref="paper", yref="paper", x=0.5, y=0.5)
                return fig
            
            # Create Gantt-style chart
            fig = go.Figure()
            
            # Color mapping for priorities
            priority_colors = {
                'Critical': '#d62728',
                'High': '#ff7f0e', 
                'Normal': '#2ca02c'
            }
            
            # Convert date strings to datetime
            schedule_df_copy = schedule_df.copy()
            schedule_df_copy['Scheduled Date'] = pd.to_datetime(schedule_df_copy['Scheduled Date'])
            
            # Sort by date
            schedule_df_copy = schedule_df_copy.sort_values('Scheduled Date')
            
            for i, row in schedule_df_copy.iterrows():
                color = priority_colors.get(row['Priority'], '#1f77b4')
                
                fig.add_trace(go.Bar(
                    x=[row['Estimated Duration']],
                    y=[f"Tool {row['Tool Id']}"],
                    orientation='h',
                    name=row['Priority'],
                    marker_color=color,
                    text=f"{row['Priority']} - {row['Maintenance Type'].title()}",
                    textposition='inside',
                    hovertemplate='<b>Tool %{y}</b><br>' +
                                  f'Date: {row["Scheduled Date"].strftime("%Y-%m-%d")}<br>' +
                                  f'Priority: {row["Priority"]}<br>' +
                                  f'Duration: {row["Estimated Duration"]} hours<br>' +
                                  f'Cost: ${row["Estimated Cost"]}<br>' +
                                  '<extra></extra>',
                    showlegend=False
                ))
            
            fig.update_layout(
                title="Maintenance Schedule Timeline",
                xaxis_title="Duration (hours)",
                yaxis_title="Tools",
                height=max(400, len(schedule_df_copy) * 30),
                barmode='overlay'
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating timeline: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
    
    def create_confusion_matrix(self, cm):
        """Create confusion matrix heatmap"""
        try:
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Model Confusion Matrix"
            )
            
            fig.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating confusion matrix: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
    
    def create_wear_distribution(self, data):
        """Create wear level distribution chart"""
        try:
            fig = px.histogram(
                data,
                x='wear_level',
                nbins=20,
                title="Tool Wear Level Distribution",
                labels={'wear_level': 'Wear Level', 'count': 'Number of Tools'},
                color_discrete_sequence=[self.color_palette['primary']]
            )
            
            # Add vertical lines for thresholds
            fig.add_vline(x=0.6, line_dash="dash", line_color="orange", 
                         annotation_text="Warning")
            fig.add_vline(x=0.8, line_dash="dash", line_color="red", 
                         annotation_text="Critical")
            
            fig.update_layout(height=400)
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating distribution: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
    
    def create_correlation_heatmap(self, data):
        """Create correlation matrix heatmap"""
        try:
            # Select numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            correlation_matrix = data[numeric_cols].corr()
            
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu",
                title="Feature Correlation Matrix"
            )
            
            fig.update_layout(height=500)
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating heatmap: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
    
    def create_real_time_gauge(self, value, title, max_value=1.0, thresholds=None):
        """Create a gauge chart for real-time monitoring"""
        try:
            if thresholds is None:
                thresholds = [0.6, 0.8]
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = value,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': title},
                delta = {'reference': thresholds[0]},
                gauge = {
                    'axis': {'range': [None, max_value]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, thresholds[0]], 'color': "lightgreen"},
                        {'range': [thresholds[0], thresholds[1]], 'color': "yellow"},
                        {'range': [thresholds[1], max_value], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': thresholds[1]
                    }
                }
            ))
            
            fig.update_layout(height=300)
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating gauge: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
