# Model Training for CNC Tool Wear Prediction

## What is Model Training?

Model training is the process of teaching a machine learning algorithm to recognize patterns in historical data so it can make accurate predictions on new, unseen data. It's similar to how students learn from textbooks and practice problems to perform well on future exams.

In our CNC tool wear prediction project, training means showing the algorithm thousands of examples of sensor readings paired with actual wear measurements, so it learns to predict wear levels from sensor data alone.

## Pre-Training Preparation

### Step 1: Data Collection and Loading

**Loading the Dataset**
```python
import pandas as pd
import numpy as np

# Load CNC sensor data from CSV file
data = pd.read_csv('cnc_tool_data.csv')

print(f"Dataset loaded: {len(data)} records from {data['tool_id'].nunique()} tools")
print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
```

**Data Quality Check**
Before training, we verify the data quality:
- Check for missing values in critical sensors
- Validate sensor readings are within realistic ranges
- Ensure timestamps are properly formatted
- Confirm wear levels are between 0 and 1

### Step 2: Data Cleaning and Preprocessing

**Remove Invalid Records**
```python
# Remove records with sensor malfunctions
initial_count = len(data)

# Remove rows where sensors show impossible values
data = data[data['vibration'] > 0]  # Vibration can't be negative
data = data[data['temperature'] > 15]  # Temperature below 15°C is unrealistic
data = data[data['cutting_force'] > 0]  # Force can't be negative

final_count = len(data)
print(f"Removed {initial_count - final_count} invalid records")
```

**Handle Missing Values**
```python
# Fill missing sensor values using interpolation
data['vibration'] = data['vibration'].fillna(method='linear')
data['temperature'] = data['temperature'].fillna(data['temperature'].mean())
data['cutting_force'] = data['cutting_force'].fillna(method='forward')
```

**Feature Engineering**
Create new features that help the model learn better:
```python
# Calculate power consumption estimate
data['power_estimate'] = (data['cutting_force'] * data['spindle_speed'] * data['feed_rate']) / 1000

# Temperature-vibration interaction
data['temp_vib_interaction'] = data['temperature'] * data['vibration']

# Efficiency score
data['efficiency_score'] = (data['spindle_speed'] * data['feed_rate']) / (data['vibration'] + 1)

# Normalized cutting parameters
data['normalized_speed'] = data['spindle_speed'] / data['spindle_speed'].max()
data['normalized_feed'] = data['feed_rate'] / data['feed_rate'].max()
```

### Step 3: Data Splitting Strategy

**Training and Testing Split**
```python
from sklearn.model_selection import train_test_split

# Separate features (X) and target variable (y)
feature_columns = ['vibration', 'temperature', 'cutting_force', 'spindle_speed', 
                  'feed_rate', 'power_estimate', 'temp_vib_interaction', 'efficiency_score']
X = data[feature_columns]
y = data['wear_level']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 20% for testing
    random_state=42,         # For reproducible results
    stratify=pd.cut(y, bins=3, labels=[0,1,2])  # Ensure balanced wear levels
)

print(f"Training set: {len(X_train)} records")
print(f"Testing set: {len(X_test)} records")
```

**Why We Split the Data**
- **Training Set (80%)**: Used to teach the algorithm patterns
- **Testing Set (20%)**: Used to evaluate how well it learned (like a final exam)
- **Stratified Split**: Ensures both sets have similar distributions of low, medium, and high wear levels

## Training Process for Each Algorithm

### Random Forest Training

**Configuration**
```python
from sklearn.ensemble import RandomForestRegressor

# Configure Random Forest parameters
rf_model = RandomForestRegressor(
    n_estimators=100,        # Number of decision trees
    max_depth=None,          # Let trees grow to full depth
    min_samples_split=5,     # Minimum samples needed to split a node
    min_samples_leaf=2,      # Minimum samples in each leaf
    max_features='sqrt',     # Number of features each tree considers
    random_state=42,         # For reproducible results
    n_jobs=-1               # Use all CPU cores for speed
)
```

**Training Process**
```python
import time

print("Starting Random Forest training...")
start_time = time.time()

# Train the model
rf_model.fit(X_train, y_train)

training_time = time.time() - start_time
print(f"Random Forest training completed in {training_time:.2f} seconds")
```

**What Happens During Training**
1. Algorithm creates 100 different decision trees
2. Each tree is trained on a random subset of data
3. Each tree uses a random selection of sensor features
4. Trees learn if-then rules to predict wear levels
5. Final model combines all trees for predictions

### SVM Training

**Data Preprocessing for SVM**
```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# SVM requires feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled for SVM training")
```

**Configuration and Training**
```python
# Configure SVM parameters
svm_model = SVR(
    kernel='rbf',           # Radial basis function (curved boundary)
    C=1.0,                  # Regularization strength
    gamma='scale',          # Kernel coefficient
    epsilon=0.1,            # Tolerance for stopping
    cache_size=1000         # Memory cache for speed
)

print("Starting SVM training...")
start_time = time.time()

# Train the model
svm_model.fit(X_train_scaled, y_train)

training_time = time.time() - start_time
print(f"SVM training completed in {training_time:.2f} seconds")
```

**What Happens During Training**
1. Algorithm plots all data points in multi-dimensional space
2. Searches for optimal boundary to separate different wear levels
3. Uses mathematical optimization to find best curve
4. Stores support vectors (key data points) for future predictions

### Gradient Boosting Training

**Configuration**
```python
from sklearn.ensemble import GradientBoostingRegressor

# Configure Gradient Boosting parameters
gb_model = GradientBoostingRegressor(
    n_estimators=100,        # Number of boosting rounds
    learning_rate=0.1,       # How fast to learn (slower is often better)
    max_depth=3,            # Depth of individual trees
    min_samples_split=20,    # Minimum samples to split
    min_samples_leaf=10,     # Minimum samples in leaf
    subsample=0.8,          # Fraction of samples each tree uses
    random_state=42,         # For reproducible results
    validation_fraction=0.1  # Data reserved for early stopping
)
```

**Training Process with Progress Monitoring**
```python
print("Starting Gradient Boosting training...")
start_time = time.time()

# Train with verbose output to monitor progress
gb_model.fit(X_train, y_train)

training_time = time.time() - start_time
print(f"Gradient Boosting training completed in {training_time:.2f} seconds")
```

**What Happens During Training**
1. Starts with simple initial prediction (average wear level)
2. Measures prediction errors for all training examples
3. Builds new model to correct those errors
4. Repeats process 100 times, each iteration improving accuracy
5. Combines all models for final predictions

## Model Evaluation and Validation

### Performance Metrics

**Regression Metrics**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions on test set
rf_predictions = rf_model.predict(X_test)
svm_predictions = svm_model.predict(X_test_scaled)
gb_predictions = gb_model.predict(X_test)

# Calculate performance metrics for each model
models = {
    'Random Forest': rf_predictions,
    'SVM': svm_predictions,
    'Gradient Boosting': gb_predictions
}

for model_name, predictions in models.items():
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"\n{model_name} Performance:")
    print(f"  Mean Absolute Error: {mae:.3f}")
    print(f"  Root Mean Square Error: {rmse:.3f}")
    print(f"  R² Score: {r2:.3f}")
```

**Classification Metrics (for Maintenance Decisions)**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Convert regression problem to classification (maintenance needed vs not needed)
maintenance_threshold = 0.8
y_test_class = (y_test > maintenance_threshold).astype(int)

for model_name, predictions in models.items():
    pred_class = (predictions > maintenance_threshold).astype(int)
    
    accuracy = accuracy_score(y_test_class, pred_class)
    precision = precision_score(y_test_class, pred_class, zero_division=0)
    recall = recall_score(y_test_class, pred_class, zero_division=0)
    
    print(f"\n{model_name} Classification Performance:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
```

### Cross-Validation for Robust Evaluation

**5-Fold Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

# Test each model using 5-fold cross-validation
cv_results = {}

for model_name, model in [('Random Forest', rf_model), 
                         ('Gradient Boosting', gb_model)]:
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_results[model_name] = cv_scores
    
    print(f"\n{model_name} Cross-Validation:")
    print(f"  R² Scores: {cv_scores}")
    print(f"  Mean: {cv_scores.mean():.3f}")
    print(f"  Standard Deviation: {cv_scores.std():.3f}")
```

### Feature Importance Analysis

**Random Forest Feature Importance**
```python
# Get feature importance from Random Forest
feature_importance = rf_model.feature_importances_
feature_names = feature_columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nFeature Importance Ranking (Random Forest):")
for idx, row in importance_df.iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")
```

**Typical Feature Importance Results**
1. **Vibration**: 0.285 (Most important - directly indicates tool condition)
2. **Cutting Force**: 0.232 (High importance - worn tools need more force)
3. **Temperature**: 0.201 (Important - heat indicates inefficient cutting)
4. **Power Estimate**: 0.156 (Moderate - combines multiple factors)
5. **Spindle Speed**: 0.126 (Lower - operating parameter, less predictive)

## Hyperparameter Tuning

### Grid Search for Optimal Parameters

**Random Forest Tuning**
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid to search
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search_rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

print("Starting hyperparameter tuning for Random Forest...")
grid_search_rf.fit(X_train, y_train)

print(f"Best parameters: {grid_search_rf.best_params_}")
print(f"Best cross-validation score: {grid_search_rf.best_score_:.3f}")
```

**SVM Tuning**
```python
# Define parameter grid for SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'epsilon': [0.01, 0.1, 0.2]
}

# Perform grid search
grid_search_svm = GridSearchCV(
    SVR(kernel='rbf'),
    param_grid_svm,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

print("Starting hyperparameter tuning for SVM...")
grid_search_svm.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search_svm.best_params_}")
print(f"Best cross-validation score: {grid_search_svm.best_score_:.3f}")
```

## Model Saving and Loading

### Saving Trained Models
```python
import joblib

# Save all trained models
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(gb_model, 'gradient_boosting_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')  # Save scaler for SVM

print("All models saved successfully")
```

### Loading Models for Production
```python
# Load saved models
rf_model_loaded = joblib.load('random_forest_model.pkl')
svm_model_loaded = joblib.load('svm_model.pkl')
gb_model_loaded = joblib.load('gradient_boosting_model.pkl')
scaler_loaded = joblib.load('feature_scaler.pkl')

print("Models loaded and ready for production use")
```

## Training Time and Resource Requirements

### Expected Training Times

| Algorithm | 1,000 Records | 10,000 Records | 100,000 Records |
|-----------|---------------|----------------|-----------------|
| Random Forest | 2-5 seconds | 15-30 seconds | 2-5 minutes |
| SVM | 5-15 seconds | 2-5 minutes | 15-45 minutes |
| Gradient Boosting | 10-30 seconds | 5-15 minutes | 30-90 minutes |

### Hardware Requirements

**Minimum Requirements**
- CPU: 4 cores, 2.0 GHz
- RAM: 8 GB
- Storage: 10 GB free space

**Recommended Requirements**
- CPU: 8+ cores, 3.0+ GHz
- RAM: 16+ GB
- Storage: 50+ GB SSD
- GPU: Optional, can accelerate some algorithms

### Memory Usage During Training

**Random Forest**: 500 MB - 2 GB (depending on number of trees)
**SVM**: 100 MB - 1 GB (depending on dataset size)
**Gradient Boosting**: 200 MB - 1 GB (depending on iterations)

## Common Training Issues and Solutions

### Issue 1: Poor Model Performance

**Symptoms**: Low accuracy, high error rates
**Possible Causes**:
- Insufficient training data
- Poor data quality
- Wrong algorithm choice
- Need feature engineering

**Solutions**:
- Collect more diverse training data
- Improve data cleaning procedures
- Try different algorithms
- Create new features from existing sensors

### Issue 2: Overfitting

**Symptoms**: High training accuracy but poor test accuracy
**Solutions**:
- Reduce model complexity (fewer trees, lower depth)
- Use cross-validation
- Collect more training data
- Apply regularization techniques

### Issue 3: Slow Training

**Symptoms**: Training takes much longer than expected
**Solutions**:
- Reduce dataset size for initial experiments
- Use fewer algorithm iterations
- Upgrade hardware
- Optimize data preprocessing

### Issue 4: Memory Errors

**Symptoms**: Training crashes due to insufficient memory
**Solutions**:
- Process data in smaller batches
- Reduce number of features
- Use algorithms with lower memory requirements
- Upgrade system RAM

This comprehensive training guide ensures your CNC tool wear prediction models are properly trained, validated, and ready for production deployment in real manufacturing environments.