import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def determine_tool_status(wear_level):
    """Determine tool status based on wear level"""
    if wear_level < 0.3:
        return {
            'status': 'GOOD',
            'color': 'GREEN',
            'action': 'Continue production',
            'alert_level': 'NONE'
        }
    elif wear_level < 0.6:
        return {
            'status': 'FAIR', 
            'color': 'YELLOW',
            'action': 'Monitor closely',
            'alert_level': 'LOW'
        }
    elif wear_level < 0.8:
        return {
            'status': 'WORN',
            'color': 'ORANGE', 
            'action': 'Schedule replacement soon',
            'alert_level': 'MEDIUM'
        }
    else:
        return {
            'status': 'CRITICAL',
            'color': 'RED',
            'action': 'Replace immediately',
            'alert_level': 'HIGH'
        }

def estimate_remaining_life(wear_level, current_runtime=0):
    """Estimate remaining tool life in hours"""
    # Simple linear extrapolation
    # Assumes tool will reach 100% wear at some point
    if wear_level >= 1.0:
        return 0.0
    
    # Estimate based on current wear rate
    # This is a simplified calculation
    max_life = 100  # Assumed maximum tool life in hours
    used_life = wear_level * max_life
    remaining = max_life - used_life
    
    return max(0, remaining)

def validate_sensor_data(data):
    """Validate sensor data ranges"""
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Define acceptable ranges
    ranges = {
        'vibration': (0.1, 20.0),
        'temperature': (10, 150),
        'cutting_force': (10, 2000),
        'spindle_speed': (100, 10000),
        'feed_rate': (10, 1000),
        'wear_level': (0.0, 1.0)
    }
    
    for column, (min_val, max_val) in ranges.items():
        if column in data.columns:
            out_of_range = data[(data[column] < min_val) | (data[column] > max_val)]
            
            if len(out_of_range) > 0:
                validation_results['warnings'].append(
                    f"{column}: {len(out_of_range)} values out of range [{min_val}, {max_val}]"
                )
    
    # Check for missing values
    missing_data = data.isnull().sum()
    for column, missing_count in missing_data.items():
        if missing_count > 0:
            validation_results['warnings'].append(
                f"{column}: {missing_count} missing values"
            )
    
    # Set validation status
    validation_results['valid'] = len(validation_results['errors']) == 0
    
    return validation_results

def generate_maintenance_schedule(data, model, threshold=0.8, days_ahead=14):
    """Generate maintenance schedule based on predictions"""
    if not model.is_trained:
        return pd.DataFrame()
    
    try:
        # Get unique tools
        tools = data['tool_id'].unique()
        maintenance_tasks = []
        
        for tool_id in tools:
            # Get latest data for this tool
            tool_data = data[data['tool_id'] == tool_id].tail(1)
            
            if len(tool_data) == 0:
                continue
            
            # Predict current wear level
            try:
                wear_level = model.predict(tool_data)[0]
            except:
                continue
            
            # Schedule maintenance if above threshold
            if wear_level > threshold:
                priority = 'Critical' if wear_level > 0.9 else 'High' if wear_level > 0.8 else 'Normal'
                
                # Estimate when to schedule (simplified)
                urgency = min(wear_level * 10, 7)  # Days from now
                scheduled_date = datetime.now() + timedelta(days=urgency)
                
                task = {
                    'Tool Id': tool_id,
                    'Current Wear': f"{wear_level:.1%}",
                    'Priority': priority,
                    'Scheduled Date': scheduled_date.strftime('%Y-%m-%d'),
                    'Maintenance Type': 'replacement',
                    'Estimated Duration': np.random.choice([2, 4, 6, 8]),
                    'Estimated Cost': np.random.choice([200, 500, 800, 1200]),
                    'Machine Id': tool_data['machine_id'].iloc[0] if 'machine_id' in tool_data.columns else 'Unknown'
                }
                
                maintenance_tasks.append(task)
        
        # Sort by priority and date
        df = pd.DataFrame(maintenance_tasks)
        if len(df) > 0:
            priority_order = {'Critical': 0, 'High': 1, 'Normal': 2}
            df['priority_sort'] = df['Priority'].map(priority_order)
            df = df.sort_values(['priority_sort', 'Scheduled Date']).drop('priority_sort', axis=1)
        
        return df
        
    except Exception as e:
        return pd.DataFrame()

def calculate_cost_savings(maintenance_schedule):
    """Calculate potential cost savings from predictive maintenance"""
    if len(maintenance_schedule) == 0:
        return {
            'preventive_cost': 0,
            'reactive_cost': 0,
            'savings': 0,
            'savings_percentage': 0
        }
    
    # Estimated costs
    preventive_cost = maintenance_schedule['Estimated Cost'].sum()
    
    # Reactive maintenance typically costs 3-5x more
    reactive_multiplier = 4
    reactive_cost = preventive_cost * reactive_multiplier
    
    # Downtime costs
    downtime_cost_per_hour = 5000  # Estimated downtime cost
    preventive_downtime = maintenance_schedule['Estimated Duration'].sum()
    reactive_downtime = preventive_downtime * 2  # Reactive takes longer
    
    total_preventive = preventive_cost + (preventive_downtime * downtime_cost_per_hour)
    total_reactive = reactive_cost + (reactive_downtime * downtime_cost_per_hour)
    
    savings = total_reactive - total_preventive
    savings_percentage = (savings / total_reactive) * 100 if total_reactive > 0 else 0
    
    return {
        'preventive_cost': total_preventive,
        'reactive_cost': total_reactive,
        'savings': savings,
        'savings_percentage': savings_percentage
    }

def export_to_csv(data, filename):
    """Export data to CSV format"""
    try:
        csv_string = data.to_csv(index=False)
        return csv_string
    except Exception as e:
        return None
