# CNC Tool Wear Prediction Dataset Documentation

## Dataset Overview

This dataset contains realistic CNC machining data for tool wear prediction and maintenance scheduling. The data simulates real-world manufacturing conditions across multiple tools, machines, and operating parameters.

## Problem Definition

### Business Problem
Manufacturing companies lose millions annually due to:
- Unplanned CNC tool failures causing production shutdowns
- Over-conservative maintenance schedules leading to premature tool replacement
- Quality issues from worn tools producing defective parts
- Lack of data-driven insights for maintenance optimization

### Technical Problem
Predict tool wear levels using sensor data and operating parameters to enable:
- Proactive maintenance scheduling
- Optimal tool replacement timing
- Quality assurance through condition monitoring
- Cost reduction through efficient tool utilization

## Data Schema

### Core Features (Required)

| Column | Type | Range | Unit | Description |
|--------|------|-------|------|-------------|
| `tool_id` | String | T001-T020 | - | Unique tool identifier |
| `timestamp` | DateTime | 2024-01-01 to present | - | Data collection time |
| `vibration` | Float | 0.5-10.0 | mm/s | Tool vibration amplitude |
| `temperature` | Float | 20-100 | °C | Cutting zone temperature |
| `cutting_force` | Float | 100-1000 | N | Force applied during cutting |
| `spindle_speed` | Integer | 500-5000 | RPM | Spindle rotation speed |
| `feed_rate` | Integer | 50-500 | mm/min | Material feed velocity |
| `wear_level` | Float | 0.0-1.0 | - | Normalized tool wear (target) |

### Optional Features (Enhanced Analysis)

| Column | Type | Range | Unit | Description |
|--------|------|-------|------|-------------|
| `machine_id` | String | CNC01-CNC05 | - | Machine identifier |
| `operation_type` | String | roughing, finishing, drilling, milling | - | Machining operation |
| `material_hardness` | String | soft, medium, hard | - | Workpiece material type |
| `cutting_time` | Float | 1-50 | hours | Cumulative cutting time |

### Derived Features (Auto-generated)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `estimated_power` | (force × feed_rate × spindle_speed) / 1000 | Power consumption estimate |
| `temp_vib_product` | temperature × vibration | Temperature-vibration interaction |
| `normalized_speed` | spindle_speed / max(spindle_speed) | Relative speed factor |
| `normalized_feed` | feed_rate / max(feed_rate) | Relative feed factor |
| `wear_rate` | diff(wear_level) / diff(timestamp) | Wear progression rate |
| `efficiency_score` | (normalized_speed × normalized_feed) / (vibration + 1) | Operating efficiency |

## Data Generation Logic

### Realistic Correlations
1. **Higher spindle speeds** → increased temperature and vibration
2. **Increased cutting forces** → higher temperatures and wear
3. **Longer cutting times** → progressive wear accumulation
4. **Hard materials** → increased forces and faster wear
5. **Optimal parameters** → lower wear rates and better efficiency

### Wear Level Calculation
```python
wear_factor = (
    (cutting_force - 100) / 900 * 0.3 +    # Force contribution (30%)
    (temperature - 20) / 80 * 0.2 +        # Temperature contribution (20%)
    (spindle_speed - 500) / 4500 * 0.2 +   # Speed contribution (20%)
    (vibration - 0.5) / 9.5 * 0.3          # Vibration contribution (30%)
)
wear_level = wear_factor + random_noise
```

### Distribution Targets
- **Low wear (0.0-0.4)**: 40% of data points
- **Medium wear (0.4-0.7)**: 35% of data points  
- **High wear (0.7-1.0)**: 25% of data points

## Sample Data Records

```csv
tool_id,timestamp,vibration,temperature,cutting_force,spindle_speed,feed_rate,wear_level,machine_id,operation_type,material_hardness,cutting_time
T001,2024-01-15 09:30:00,2.3,45.2,420.5,2800,180,0.25,CNC01,finishing,medium,12.5
T001,2024-01-15 10:30:00,2.8,48.1,445.2,2850,185,0.28,CNC01,finishing,medium,13.5
T002,2024-01-15 09:31:00,3.1,52.8,580.2,3200,220,0.68,CNC02,roughing,hard,28.2
T003,2024-01-15 09:32:00,1.8,42.3,380.1,2400,160,0.15,CNC03,drilling,soft,8.1
T004,2024-01-15 09:33:00,4.2,65.4,720.8,3800,280,0.85,CNC04,milling,hard,35.7
```

## Data Quality Standards

### Completeness Requirements
- **Missing values**: <5% for core features
- **Timestamp gaps**: No gaps >4 hours in production data
- **Tool coverage**: Minimum 10 different tools
- **Temporal span**: At least 30 days of historical data

### Accuracy Specifications
- **Sensor calibration**: ±2% accuracy for all measurements
- **Synchronization**: All sensors timestamped within ±1 second
- **Range validation**: All values within specified operational ranges
- **Outlier detection**: Automated flagging of values >3 standard deviations

### Consistency Checks
- **Monotonic wear**: Wear levels should not decrease over time for same tool
- **Physical constraints**: Temperature cannot exceed material melting point
- **Operating limits**: Speeds and feeds within machine specifications
- **Correlation validation**: Expected relationships between parameters maintained

## Usage Instructions

### For Model Training
1. **Load data**: Use the sample data generator or upload your CSV file
2. **Validate quality**: Review data statistics and missing value reports
3. **Feature engineering**: System automatically creates derived features
4. **Split data**: 80% training, 20% testing (stratified by wear level)
5. **Train models**: Compare Random Forest, SVM, and Gradient Boosting

### For Real-time Prediction
1. **Input current parameters**: Vibration, temperature, force, speed, feed rate
2. **Get predictions**: Wear level, confidence score, remaining life estimate
3. **Monitor thresholds**: Green (<0.6), Yellow (0.6-0.8), Red (>0.8)
4. **Schedule maintenance**: Based on predicted wear progression

### For Maintenance Planning
1. **Set thresholds**: Configure wear levels for different priority levels
2. **Specify horizon**: Plan maintenance 7-90 days ahead
3. **Optimize schedule**: Balance workload across maintenance days
4. **Export calendar**: Download schedule for integration with ERP systems

## Integration Examples

### CSV Upload Format
```csv
tool_id,vibration,temperature,cutting_force,spindle_speed,feed_rate,wear_level,timestamp
T001,2.5,45.0,400.0,2500,200,0.30,2024-01-15T09:00:00
```

### API Integration (Future)
```json
{
  "tool_id": "T001",
  "timestamp": "2024-01-15T09:00:00Z",
  "sensors": {
    "vibration": 2.5,
    "temperature": 45.0,
    "cutting_force": 400.0,
    "spindle_speed": 2500,
    "feed_rate": 200
  }
}
```

### Database Schema (SQL)
```sql
CREATE TABLE cnc_tool_data (
    id SERIAL PRIMARY KEY,
    tool_id VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    vibration DECIMAL(4,2),
    temperature DECIMAL(5,2),
    cutting_force DECIMAL(6,2),
    spindle_speed INTEGER,
    feed_rate INTEGER,
    wear_level DECIMAL(4,3),
    machine_id VARCHAR(10),
    operation_type VARCHAR(20)
);
```

## Performance Benchmarks

### Model Accuracy Targets
- **Precision**: >85% for high wear detection
- **Recall**: >90% for critical wear levels
- **R² Score**: >0.80 for wear level regression
- **Prediction Speed**: <100ms per inference

### Business KPIs
- **Downtime Reduction**: 35-50% improvement
- **Tool Cost Savings**: 25-40% reduction in premature replacement
- **Quality Improvement**: 60% fewer defective parts
- **Maintenance Efficiency**: 45% better resource allocation

This comprehensive dataset enables accurate tool wear prediction and optimal maintenance scheduling for CNC manufacturing operations.