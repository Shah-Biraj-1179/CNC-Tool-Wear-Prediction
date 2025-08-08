import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_sample_data(n_records=1000, n_tools=20):
    """Generate sample CNC tool wear data for demonstration"""

    np.random.seed(42)  # For reproducible results

    # Generate tool IDs
    tool_ids = [f"T{i:03d}" for i in range(1, n_tools + 1)]

    # Generate timestamps
    start_date = datetime.now() - timedelta(days=30)

    data_records = []

    for i in range(n_records):
        # Select random tool
        tool_id = np.random.choice(tool_ids)

        # Generate timestamp (more recent data more likely)
        days_back = np.random.exponential(
            5)  # Exponential distribution for realistic timestamps
        timestamp = start_date + timedelta(days=min(days_back, 30))

        # Generate base parameters with some correlation

        # Spindle speed affects other parameters
        spindle_speed = np.random.normal(2500, 500)
        spindle_speed = max(500, min(5000, spindle_speed))

        # Feed rate correlated with spindle speed
        feed_rate = np.random.normal(200 + (spindle_speed - 2500) * 0.05, 50)
        feed_rate = max(50, min(500, feed_rate))

        # Cutting force influenced by feed rate and spindle speed
        base_force = 400 + (feed_rate - 200) * 2 + (spindle_speed - 2500) * 0.1
        cutting_force = np.random.normal(base_force, 100)
        cutting_force = max(100, min(1000, cutting_force))

        # Temperature influenced by spindle speed and cutting force
        base_temp = 45 + (spindle_speed - 2500) * 0.005 + (cutting_force -
                                                           400) * 0.02
        temperature = np.random.normal(base_temp, 10)
        temperature = max(20, min(100, temperature))

        # Vibration influenced by wear and operating conditions
        base_vibration = 2.5 + (spindle_speed -
                                2500) * 0.001 + (cutting_force - 400) * 0.002
        vibration = np.random.normal(base_vibration, 0.5)
        vibration = max(0.5, min(10, vibration))

        # Calculate wear level based on operating conditions and some randomness
        # Higher forces, temperatures, speeds generally lead to more wear
        wear_factor = (
            (cutting_force - 100) / 900 * 0.3 +  # Force contribution
            (temperature - 20) / 80 * 0.2 +  # Temperature contribution
            (spindle_speed - 500) / 4500 * 0.2 +  # Speed contribution
            (vibration - 0.5) / 9.5 * 0.3  # Vibration contribution
        )

        # Add some randomness and ensure wear is realistic
        wear_level = wear_factor + np.random.normal(0, 0.1)
        wear_level = max(0.05, min(0.95, wear_level))  # Realistic wear bounds

        # Some tools should have higher wear (end of life)
        if np.random.random() < 0.1:  # 10% chance of high wear tool
            wear_level = np.random.uniform(0.8, 0.95)

        record = {
            'tool_id':
            tool_id,
            'timestamp':
            timestamp,
            'vibration':
            round(vibration, 2),
            'temperature':
            round(temperature, 1),
            'cutting_force':
            round(cutting_force, 1),
            'spindle_speed':
            int(spindle_speed),
            'feed_rate':
            int(feed_rate),
            'wear_level':
            round(wear_level, 3),
            'machine_id':
            f"CNC{np.random.randint(1, 6):02d}",  # 5 machines
            'operation_type':
            np.random.choice(['roughing', 'finishing', 'drilling', 'milling'])
        }

        data_records.append(record)

    # Create DataFrame
    df = pd.DataFrame(data_records)

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Add some derived features that might be useful
    df['cutting_time'] = np.random.uniform(1, 50,
                                           len(df))  # Hours of cutting time
    df['material_hardness'] = np.random.choice(['soft', 'medium', 'hard'],
                                               len(df))

    # Ensure we have some variety in wear levels
    wear_distribution = {
        'low': (0.05, 0.4),
        'medium': (0.4, 0.7),
        'high': (0.7, 0.95)
    }

    # Adjust distribution to ensure we have examples in each category
    n_each = len(df) // 3

    for i, (category, (min_wear,
                       max_wear)) in enumerate(wear_distribution.items()):
        start_idx = i * n_each
        end_idx = start_idx + n_each if i < 2 else len(df)

        for idx in range(start_idx, min(end_idx, len(df))):
            df.loc[idx, 'wear_level'] = np.random.uniform(min_wear, max_wear)

    # Round wear levels to 3 decimal places
    df['wear_level'] = df['wear_level'].round(3)

    return df


def get_sample_maintenance_history():
    """Generate sample maintenance history data"""

    np.random.seed(42)

    # Generate 50 historical maintenance records
    maintenance_records = []

    for i in range(50):
        record = {
            'maintenance_id':
            f"MAINT{i+1:03d}",
            'tool_id':
            f"T{np.random.randint(1, 21):03d}",
            'maintenance_date':
            datetime.now() - timedelta(days=np.random.randint(1, 365)),
            'maintenance_type':
            np.random.choice(['preventive', 'corrective', 'emergency'],
                             p=[0.6, 0.3, 0.1]),
            'duration_hours':
            np.random.choice([2, 4, 8, 24], p=[0.4, 0.3, 0.2, 0.1]),
            'cost':
            np.random.choice([100, 300, 500, 2000], p=[0.4, 0.3, 0.2, 0.1]),
            'technician':
            f"Tech{np.random.randint(1, 6):02d}",
            'downtime_hours':
            np.random.choice([1, 2, 4, 8], p=[0.4, 0.3, 0.2, 0.1]),
            'effectiveness':
            np.random.uniform(0.8, 1.0)
        }

        maintenance_records.append(record)

    return pd.DataFrame(maintenance_records)


def get_real_time_sample():
    """Generate a single record for real-time demonstration"""

    return {
        'tool_id': 'T001',
        'timestamp': datetime.now(),
        'vibration': np.random.uniform(1.5, 4.0),
        'temperature': np.random.uniform(40, 60),
        'cutting_force': np.random.uniform(300, 500),
        'spindle_speed': np.random.randint(2000, 3000),
        'feed_rate': np.random.randint(150, 250),
        'cutting_time': 25.5,
        'machine_id': 'CNC01',
        'operation_type': 'finishing'
    }
