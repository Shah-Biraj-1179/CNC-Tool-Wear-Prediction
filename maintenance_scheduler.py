import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class MaintenanceScheduler:
    def __init__(self):
        self.maintenance_types = {
            'preventive': {'duration': 2, 'cost': 100, 'effectiveness': 0.9},
            'corrective': {'duration': 8, 'cost': 500, 'effectiveness': 1.0},
            'emergency': {'duration': 24, 'cost': 2000, 'effectiveness': 1.0}
        }
        
        self.priority_thresholds = {
            'critical': 0.9,
            'high': 0.8,
            'normal': 0.6
        }
    
    def generate_schedule(self, data, model, days_ahead=30, threshold=0.8, buffer_days=3):
        """Generate maintenance schedule based on predictions"""
        try:
            schedule_list = []
            
            # Get unique tools
            tools = data['tool_id'].unique()
            
            for tool_id in tools:
                # Get latest data for this tool
                tool_data = data[data['tool_id'] == tool_id].tail(1)
                
                if tool_data.empty:
                    continue
                
                # Make prediction for current state
                current_wear = model.predict(tool_data)[0]
                
                # Calculate wear progression rate
                tool_history = data[data['tool_id'] == tool_id].sort_values('timestamp')
                if len(tool_history) > 1:
                    wear_history = tool_history['wear_level'].values
                    time_diff = (tool_history['timestamp'].iloc[-1] - tool_history['timestamp'].iloc[0]).days
                    if time_diff > 0:
                        wear_rate = (wear_history[-1] - wear_history[0]) / time_diff
                    else:
                        wear_rate = 0.01  # Default rate
                else:
                    wear_rate = 0.01
                
                # Predict when maintenance will be needed
                days_to_threshold = self._calculate_days_to_threshold(
                    current_wear, threshold, wear_rate
                )
                
                if days_to_threshold <= days_ahead:
                    # Determine priority and maintenance type
                    priority = self._determine_priority(current_wear)
                    maintenance_type = self._determine_maintenance_type(current_wear)
                    
                    # Schedule maintenance with buffer
                    scheduled_date = datetime.now() + timedelta(days=max(1, days_to_threshold - buffer_days))
                    
                    # Get tool details
                    latest_record = tool_data.iloc[0]
                    
                    schedule_entry = {
                        'tool_id': tool_id,
                        'scheduled_date': scheduled_date.strftime('%Y-%m-%d'),
                        'current_wear': current_wear,
                        'predicted_wear_at_maintenance': min(1.0, current_wear + (wear_rate * days_to_threshold)),
                        'days_until_maintenance': int(days_to_threshold),
                        'priority': priority,
                        'maintenance_type': maintenance_type,
                        'estimated_duration': self.maintenance_types[maintenance_type]['duration'],
                        'estimated_cost': self.maintenance_types[maintenance_type]['cost'],
                        'current_vibration': latest_record['vibration'],
                        'current_temperature': latest_record['temperature'],
                        'current_force': latest_record['cutting_force'],
                        'recommended_action': self._get_recommended_action(current_wear, maintenance_type),
                        'risk_level': self._calculate_risk_level(current_wear, days_to_threshold)
                    }
                    
                    schedule_list.append(schedule_entry)
            
            # Convert to DataFrame and sort by priority and date
            if schedule_list:
                schedule_df = pd.DataFrame(schedule_list)
                
                # Sort by priority (Critical first) then by scheduled date
                priority_order = {'Critical': 0, 'High': 1, 'Normal': 2}
                schedule_df['priority_rank'] = schedule_df['priority'].apply(lambda x: priority_order.get(x, 2))
                schedule_df = schedule_df.sort_values(['priority_rank', 'scheduled_date']).drop('priority_rank', axis=1)
                
                # Capitalize column names for better display
                schedule_df.columns = [col.replace('_', ' ').title() for col in schedule_df.columns]
                
                return schedule_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            raise Exception(f"Schedule generation failed: {str(e)}")
    
    def _calculate_days_to_threshold(self, current_wear, threshold, wear_rate):
        """Calculate days until wear reaches threshold"""
        if current_wear >= threshold:
            return 0  # Immediate maintenance needed
        
        if wear_rate <= 0:
            return 999  # Very slow wear
        
        days = (threshold - current_wear) / wear_rate
        return max(1, int(days))
    
    def _determine_priority(self, current_wear):
        """Determine maintenance priority based on current wear"""
        if current_wear >= self.priority_thresholds['critical']:
            return 'Critical'
        elif current_wear >= self.priority_thresholds['high']:
            return 'High'
        else:
            return 'Normal'
    
    def _determine_maintenance_type(self, current_wear):
        """Determine type of maintenance needed"""
        if current_wear >= 0.9:
            return 'emergency'
        elif current_wear >= 0.8:
            return 'corrective'
        else:
            return 'preventive'
    
    def _get_recommended_action(self, current_wear, maintenance_type):
        """Get recommended maintenance action"""
        actions = {
            'preventive': [
                "Inspect tool condition",
                "Clean and lubricate",
                "Check alignment",
                "Replace consumables"
            ],
            'corrective': [
                "Replace cutting edge",
                "Recalibrate machine",
                "Check tool holder",
                "Inspect spindle condition"
            ],
            'emergency': [
                "Stop operation immediately",
                "Replace tool completely",
                "Inspect machine damage",
                "Full system check required"
            ]
        }
        
        action_list = actions.get(maintenance_type, actions['preventive'])
        
        if current_wear >= 0.95:
            return "URGENT: " + action_list[0]
        else:
            return action_list[0]  # Return primary recommended action
    
    def _calculate_risk_level(self, current_wear, days_until_maintenance):
        """Calculate risk level for the tool"""
        if current_wear >= 0.9 or days_until_maintenance <= 1:
            return "Very High"
        elif current_wear >= 0.8 or days_until_maintenance <= 3:
            return "High"
        elif current_wear >= 0.6 or days_until_maintenance <= 7:
            return "Medium"
        else:
            return "Low"
    
    def optimize_maintenance_schedule(self, schedule_df, max_daily_maintenance=3):
        """Optimize maintenance schedule to balance workload"""
        if schedule_df.empty:
            return schedule_df
        
        try:
            optimized_schedule = schedule_df.copy()
            
            # Group by scheduled date
            daily_counts = schedule_df.groupby('Scheduled Date').size()
            
            # Redistribute maintenance if too many on same day
            for date, count in daily_counts.items():
                if count > max_daily_maintenance:
                    # Get tools scheduled for this date
                    same_date_tools = schedule_df[schedule_df['Scheduled Date'] == date]
                    
                    # Keep highest priority ones on original date
                    priority_order = {'Critical': 0, 'High': 1, 'Normal': 2}
                    same_date_tools['priority_rank'] = same_date_tools['Priority'].map(priority_order)
                    same_date_tools = same_date_tools.sort_values('priority_rank')
                    
                    # Keep top priority tools on original date
                    keep_original = same_date_tools.head(max_daily_maintenance)
                    reschedule = same_date_tools.tail(count - max_daily_maintenance)
                    
                    # Reschedule others to nearby dates
                    for idx, row in reschedule.iterrows():
                        original_date = pd.to_datetime(date)
                        # Try to schedule 1-3 days later
                        for offset in range(1, 4):
                            new_date = original_date + timedelta(days=offset)
                            new_date_str = new_date.strftime('%Y-%m-%d')
                            
                            # Check if new date doesn't exceed capacity
                            if daily_counts.get(new_date_str, 0) < max_daily_maintenance:
                                optimized_schedule.loc[idx, 'Scheduled Date'] = new_date_str
                                daily_counts[new_date_str] = daily_counts.get(new_date_str, 0) + 1
                                break
            
            return optimized_schedule
            
        except Exception as e:
            # If optimization fails, return original schedule
            return schedule_df
    
    def calculate_maintenance_cost(self, schedule_df):
        """Calculate total maintenance cost for the schedule"""
        if schedule_df.empty:
            return 0
        
        try:
            total_cost = 0
            
            for _, row in schedule_df.iterrows():
                maintenance_type = row.get('Maintenance Type', 'preventive').lower()
                cost = self.maintenance_types.get(maintenance_type, {}).get('cost', 100)
                total_cost += cost
            
            return total_cost
            
        except Exception:
            return 0
    
    def export_schedule_to_calendar(self, schedule_df, format='ical'):
        """Export maintenance schedule to calendar format"""
        # This would typically generate iCal or other calendar format
        # For now, return a formatted string
        
        if schedule_df.empty:
            return ""
        
        calendar_entries = []
        
        for _, row in schedule_df.iterrows():
            entry = f"""
BEGIN:VEVENT
DTSTART:{row['Scheduled Date'].replace('-', '')}T090000Z
DTEND:{row['Scheduled Date'].replace('-', '')}T170000Z
SUMMARY:Tool {row['Tool Id']} Maintenance - {row['Priority']} Priority
DESCRIPTION:Maintenance Type: {row['Maintenance Type']}\\nCurrent Wear: {row['Current Wear']:.2f}\\nRecommended Action: {row['Recommended Action']}
END:VEVENT
"""
            calendar_entries.append(entry)
        
        return "\n".join(calendar_entries)
