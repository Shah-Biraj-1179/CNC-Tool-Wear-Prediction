import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class DataProcessor:
    def __init__(self):
        self.required_columns = [
            'tool_id', 'vibration', 'temperature', 'cutting_force',
            'spindle_speed', 'feed_rate', 'wear_level'
        ]
        self.optional_columns = ['timestamp', 'machine_id', 'operation_type']
        
    def validate_and_clean_data(self, data):
        """Validate and clean uploaded CNC data"""
        try:
            df = data.copy()
            
            # Check required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Data type validation and conversion
            numeric_columns = ['vibration', 'temperature', 'cutting_force', 
                             'spindle_speed', 'feed_rate', 'wear_level']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with invalid numeric data
            initial_count = len(df)
            df = df.dropna(subset=numeric_columns)
            removed_count = initial_count - len(df)
            
            if removed_count > 0:
                logging.warning(f"Removed {removed_count} rows with invalid numeric data")
            
            # Validate data ranges
            df = self._validate_sensor_ranges(df)
            
            # Add timestamp if not present
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.date_range(
                    start=datetime.now() - timedelta(hours=len(df)),
                    periods=len(df),
                    freq='H'
                )
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Feature engineering
            df = self._add_derived_features(df)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['tool_id', 'timestamp'], keep='last')
            
            return df
            
        except Exception as e:
            raise Exception(f"Data validation failed: {str(e)}")
    
    def _validate_sensor_ranges(self, df):
        """Validate sensor reading ranges"""
        # Define reasonable ranges for CNC parameters
        ranges = {
            'vibration': (0, 20),          # mm/s
            'temperature': (15, 150),       # °C
            'cutting_force': (50, 2000),   # N
            'spindle_speed': (100, 10000),  # RPM
            'feed_rate': (10, 1000),       # mm/min
            'wear_level': (0, 1)           # normalized 0-1
        }
        
        for column, (min_val, max_val) in ranges.items():
            if column in df.columns:
                # Flag outliers but don't remove them - just cap extreme values
                df[column] = df[column].clip(lower=min_val * 0.5, upper=max_val * 1.5)
        
        return df
    
    def _add_derived_features(self, df):
        """Add derived features for better model performance"""
        try:
            # Power estimation
            df['estimated_power'] = (df['cutting_force'] * df['feed_rate'] * df['spindle_speed']) / 1000
            
            # Temperature-vibration interaction
            df['temp_vib_product'] = df['temperature'] * df['vibration']
            
            # Normalized cutting parameters
            df['normalized_speed'] = df['spindle_speed'] / df['spindle_speed'].max()
            df['normalized_feed'] = df['feed_rate'] / df['feed_rate'].max()
            
            # Wear rate (if we have multiple timestamps per tool)
            df['wear_rate'] = df.groupby('tool_id')['wear_level'].diff().fillna(0)
            
            # Tool usage hours (simplified calculation)
            df['usage_hours'] = df.groupby('tool_id').cumcount() + 1
            
            # Operating efficiency score
            df['efficiency_score'] = (
                df['normalized_speed'] * df['normalized_feed'] / 
                (df['vibration'] / df['vibration'].max() + 1)
            )
            
            return df
            
        except Exception as e:
            logging.warning(f"Feature engineering partially failed: {str(e)}")
            return df
    
    def prepare_training_data(self, df, target_column='wear_level'):
        """Prepare data for model training"""
        try:
            # Select features for training
            feature_columns = [
                'vibration', 'temperature', 'cutting_force', 'spindle_speed',
                'feed_rate', 'estimated_power', 'temp_vib_product',
                'usage_hours', 'efficiency_score'
            ]
            
            # Filter available features
            available_features = [col for col in feature_columns if col in df.columns]
            
            X = df[available_features].copy()
            y = df[target_column].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
            
            return X, y, available_features
            
        except Exception as e:
            raise Exception(f"Training data preparation failed: {str(e)}")
    
    def detect_anomalies(self, df):
        """Detect anomalies in sensor data"""
        anomalies = pd.DataFrame()
        
        try:
            numeric_columns = ['vibration', 'temperature', 'cutting_force', 
                             'spindle_speed', 'feed_rate']
            
            for col in numeric_columns:
                if col in df.columns:
                    # Use IQR method for anomaly detection
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    col_anomalies = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    col_anomalies['anomaly_type'] = f'{col}_outlier'
                    
                    anomalies = pd.concat([anomalies, col_anomalies])
            
            return anomalies.drop_duplicates()
            
        except Exception as e:
            logging.error(f"Anomaly detection failed: {str(e)}")
            return pd.DataFrame()
    
    def generate_data_quality_report(self, df):
        """Generate a data quality report"""
        try:
            report = {
                'total_records': len(df),
                'total_tools': df['tool_id'].nunique() if 'tool_id' in df.columns else 0,
                'date_range': {
                    'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                    'end': df['timestamp'].max() if 'timestamp' in df.columns else None
                },
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'numeric_summary': df.describe().to_dict(),
                'anomalies_detected': len(self.detect_anomalies(df))
            }
            
            return report
            
        except Exception as e:
            logging.error(f"Data quality report generation failed: {str(e)}")
            return {'error': str(e)}
    
    def export_processed_data(self, df, filepath):
        """Export processed data to CSV"""
        try:
            df.to_csv(filepath, index=False)
            return True
        except Exception as e:
            logging.error(f"Data export failed: {str(e)}")
            return False
