# Dataset for CNC Tool Wear Prediction

## What is a Dataset?

A dataset is a structured collection of information that machine learning algorithms use to learn patterns and make predictions. Think of it as a large spreadsheet where each row represents one measurement and each column represents a different piece of information.

For our CNC tool wear prediction project, the dataset contains sensor readings from manufacturing machines along with information about how worn the cutting tools are.

## Dataset Components

### Input Features (What We Measure)

#### 1. Tool Identification
- **tool_id**: Unique identifier for each cutting tool (Examples: T001, T002, T003...)
- **machine_id**: Which CNC machine is being monitored (Examples: CNC01, CNC02, CNC03...)
- **timestamp**: Exact date and time when measurement was taken (Example: 2024-01-15 09:30:00)

#### 2. Sensor Measurements

**Vibration Sensor**
- **What it measures**: How much the tool shakes while cutting (Range: 0.5-10.0 mm/s)
- **Why it's important**: Worn tools have rough, uneven cutting edges that cause more vibration
- **How it's measured**: Accelerometer sensor attached to the spindle housing
- **Example values**: 
  - New tool: 1.5-2.5 mm/s (smooth cutting)
  - Worn tool: 4.0-8.0 mm/s (rough cutting)

**Temperature Sensor**
- **What it measures**: Heat generated in the cutting zone (Range: 20-100°C)
- **Why it's important**: Worn tools create more friction, generating excess heat
- **How it's measured**: Infrared sensor or thermocouple near the cutting area
- **Example values**:
  - New tool: 35-45°C (efficient cutting)
  - Worn tool: 60-85°C (excessive friction)

**Cutting Force Sensor**
- **What it measures**: Force required to cut through material (Range: 100-1000 N)
- **Why it's important**: Dull tools require more force to cut effectively
- **How it's measured**: Force sensor built into the tool holder or spindle
- **Example values**:
  - New tool: 200-400 N (easy cutting)
  - Worn tool: 600-900 N (difficult cutting)

**Spindle Speed Encoder**
- **What it measures**: How fast the cutting tool rotates (Range: 500-5000 RPM)
- **Why it's important**: Operating parameter that affects tool wear rate
- **How it's measured**: Digital encoder attached to the spindle motor
- **Example values**: 2500 RPM for aluminum, 1200 RPM for steel

**Feed Rate Encoder**
- **What it measures**: Speed at which material moves past the tool (Range: 50-500 mm/min)
- **Why it's important**: Operating parameter that affects cutting conditions
- **How it's measured**: Linear encoder on machine movement axes
- **Example values**: 200 mm/min for finishing, 400 mm/min for roughing

#### 3. Target Variable (What We Want to Predict)

**Wear Level**
- **What it represents**: Tool wear as a percentage from 0 to 1
- **Scale**: 0.0 = brand new tool, 1.0 = completely worn out and unusable
- **How it's determined**: Physical inspection, tool life tracking, or wear measurement devices
- **Example interpretations**:
  - 0.0-0.3: Good condition (green light)
  - 0.3-0.7: Moderate wear (yellow light)
  - 0.7-1.0: High wear (red light - needs replacement)

### Optional Context Features

**Operation Type**
- **roughing**: Removing large amounts of material quickly
- **finishing**: Creating precise final dimensions and smooth surfaces
- **drilling**: Making holes in the material
- **milling**: Creating complex shapes and features

**Material Hardness**
- **soft**: Aluminum, plastic (easy to cut)
- **medium**: Steel, brass (moderate cutting difficulty)
- **hard**: Titanium, hardened steel (difficult to cut, causes fast wear)

**Cutting Time**
- **What it measures**: Total hours the tool has been actively cutting
- **Range**: 1-50 hours typically
- **Importance**: Longer cutting times generally lead to more wear

## Sample Dataset Structure

```csv
tool_id,timestamp,vibration,temperature,cutting_force,spindle_speed,feed_rate,wear_level,machine_id,operation_type,material_hardness,cutting_time
T001,2024-01-15 09:30:00,2.3,45.2,420.5,2800,180,0.25,CNC01,finishing,medium,12.5
T001,2024-01-15 09:31:00,2.4,45.8,425.2,2800,180,0.26,CNC01,finishing,medium,12.6
T002,2024-01-15 09:31:00,3.1,52.8,580.2,3200,220,0.68,CNC02,roughing,hard,28.2
T003,2024-01-15 09:32:00,1.8,42.3,380.1,2400,160,0.15,CNC03,drilling,soft,8.1
T004,2024-01-15 09:33:00,4.2,65.4,720.8,3800,280,0.85,CNC04,milling,hard,35.7
```

## Data Collection Process

### Step 1: Sensor Installation
- Mount vibration sensors on spindle housing
- Install temperature sensors near cutting zone
- Integrate force sensors into tool holders
- Connect speed encoders to motor controllers
- Set up data acquisition computers

### Step 2: Data Acquisition
- Collect sensor readings every 10-60 seconds during operation
- Store data in industrial databases (SQL Server, Oracle, etc.)
- Ensure timestamp synchronization across all sensors
- Implement data quality checks and validation

### Step 3: Data Storage
- Real-time data streaming to monitoring systems
- Historical data archived for model training
- Backup systems to prevent data loss
- Security measures to protect industrial data

## Realistic Data Generation

Since real industrial data isn't always available for training, we create realistic synthetic data using these principles:

### Physical Relationships
1. **Higher cutting forces increase temperature**: More force means more friction and heat
2. **Faster spindle speeds cause more vibration**: Higher speeds amplify any tool imbalances
3. **Worn tools require more force**: Dull cutting edges need extra pressure to cut
4. **Worn tools generate more heat**: Increased friction from poor cutting efficiency
5. **Tool wear progresses gradually**: Wear doesn't jump suddenly but increases steadily

### Mathematical Formula for Wear Level
```
wear_level = (cutting_force_factor × 0.30) + 
             (temperature_factor × 0.20) + 
             (vibration_factor × 0.30) + 
             (time_factor × 0.20) + 
             random_noise

Where each factor is normalized between 0 and 1
```

### Realistic Noise and Variations
- Add measurement uncertainty typical of industrial sensors (±2% accuracy)
- Include occasional sensor glitches and communication errors
- Simulate environmental effects like temperature changes throughout the day
- Account for different operator techniques and material variations

## Dataset Quality Requirements

### Size Requirements
- **Minimum for testing**: 1,000 records from at least 10 different tools
- **Recommended for production**: 10,000+ records from 50+ tools
- **Ideal for advanced models**: 100,000+ records across multiple machines and conditions
- **Time span**: At least 30 days of historical data, preferably 6+ months

### Quality Standards

**Completeness**
- Less than 5% missing values in critical sensor readings
- No gaps longer than 4 hours in production data
- All tools represented across their full lifecycle (new to worn out)
- Multiple examples of each operating condition

**Accuracy**
- Sensor calibration certificates showing ±2% accuracy
- Timestamp synchronization within ±1 second across all sensors
- Validation against manual measurements and inspections
- Cross-checking between related measurements (force vs. power consumption)

**Consistency**
- Sensor readings within realistic operational ranges
- Monotonic wear progression (wear level doesn't decrease over time for same tool)
- Physical relationships maintained (higher force correlates with higher temperature)
- Consistent measurement units and scales

### Data Validation Checks

**Range Validation**
```python
# Check if sensor readings are within realistic ranges
vibration_range = (0.5, 10.0)  # mm/s
temperature_range = (20, 100)   # °C
force_range = (100, 1000)       # N
speed_range = (500, 5000)       # RPM
feed_range = (50, 500)          # mm/min
wear_range = (0.0, 1.0)         # normalized
```

**Correlation Validation**
- Vibration and wear level should be positively correlated
- Temperature and cutting force should show positive correlation
- Tool wear should increase monotonically over time for each tool

**Outlier Detection**
- Flag readings more than 3 standard deviations from the mean
- Identify sensor malfunctions (constant readings, impossible values)
- Mark periods of machine maintenance or calibration

## Data Preprocessing Steps

### Cleaning
1. Remove records with sensor malfunctions
2. Fill small gaps using interpolation
3. Remove duplicate timestamps
4. Correct obvious data entry errors

### Feature Engineering
Create new features that help models learn better patterns:

**Power Estimation**
```python
power = (cutting_force × spindle_speed × feed_rate) / 1000
```

**Efficiency Score**
```python
efficiency = (spindle_speed × feed_rate) / (vibration + 1)
```

**Temperature-Vibration Interaction**
```python
temp_vib_product = temperature × vibration
```

**Wear Rate**
```python
wear_rate = change_in_wear_level / time_elapsed
```

### Normalization
- Scale all sensor readings to similar ranges (0-1 or -1 to +1)
- Standardize timestamps to consistent format
- Convert categorical variables (operation_type, material_hardness) to numerical codes

This comprehensive dataset provides the foundation for training accurate machine learning models that can predict tool wear and optimize maintenance schedules in real manufacturing environments.