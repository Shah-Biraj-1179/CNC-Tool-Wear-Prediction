# CNC Tool Wear Prediction System

## Overview

This is an AI-powered CNC Tool Wear Prediction System built with Streamlit that provides predictive maintenance capabilities for industrial CNC operations. The system uses machine learning to analyze sensor data from CNC machines and predict tool wear levels, enabling proactive maintenance scheduling to reduce downtime, optimize tool replacement timing, and improve production efficiency. The application features real-time monitoring, interactive dashboards, and intelligent maintenance scheduling with multiple machine learning algorithms including Random Forest, Support Vector Machine, and Gradient Boosting.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Framework
- **Frontend**: Streamlit-based web application with multi-page navigation architecture
- **Session Management**: Streamlit session state for maintaining application state, model instances, and data across user interactions
- **Page Structure**: Modular page-based architecture with dedicated modules for data upload, model training, predictions, maintenance scheduling, and analytics
- **Component Architecture**: Separation of concerns with distinct modules for ML models, data processing, visualization, and utility functions

### Machine Learning Pipeline
- **Multi-Model Support**: Ensemble approach supporting Random Forest (primary), Support Vector Machine, and Gradient Boosting regressors
- **Feature Engineering**: Automated feature creation including derived metrics (vibration-temperature ratio, force-speed ratio, power estimates)
- **Data Preprocessing**: Comprehensive pipeline with data validation, cleaning, outlier detection, and standardization using StandardScaler
- **Model Persistence**: Joblib-based model serialization and loading for deployment and reuse
- **Performance Evaluation**: Comprehensive metrics including MAE, RMSE, R² score with cross-validation

### Data Management Architecture
- **Sample Data Generation**: Built-in synthetic data generator creating realistic CNC operational parameters with correlated sensor readings
- **Data Validation**: Schema validation with required/optional column checking, data type enforcement, and range validation
- **Feature Schema**: Core sensor inputs (vibration, temperature, cutting_force, spindle_speed, feed_rate) plus optional metadata (machine_id, operation_type)
- **Data Processing Pipeline**: Automated cleaning, missing value handling, and time-series preparation

### Predictive Maintenance System
- **Intelligent Scheduling**: Multi-tier maintenance scheduling (preventive, corrective, emergency) with cost optimization algorithms
- **Threshold Management**: Configurable wear level thresholds for different alert and maintenance priority levels
- **Real-time Monitoring**: Continuous prediction capability with status determination and remaining life estimation
- **Cost Analysis**: Built-in cost-benefit analysis for maintenance decisions

### Visualization and Analytics
- **Interactive Dashboards**: Plotly-based visualization system with consistent theming and color schemes
- **Real-time Charts**: Dynamic trend analysis, wear level monitoring, correlation matrices, and maintenance schedule visualization
- **Analytics Engine**: Feature importance analysis, model performance visualization, and tool lifecycle analytics
- **Multi-page Interface**: Dedicated pages for different aspects (upload, training, predictions, scheduling, analytics)

## External Dependencies

### Core Framework Dependencies
- **Streamlit**: Web application framework providing the user interface and page management
- **Pandas & NumPy**: Data manipulation, numerical computing, and time-series handling
- **Plotly**: Interactive visualization library for charts, graphs, and dashboard components
- **Scikit-learn**: Machine learning library providing algorithms, preprocessing tools, and evaluation metrics
- **Joblib**: Model serialization and persistence for trained model storage

### Machine Learning Stack
- **RandomForestRegressor**: Primary ensemble method for tool wear prediction
- **SVR (Support Vector Regression)**: Alternative algorithm for non-linear pattern recognition
- **GradientBoostingRegressor**: Advanced ensemble method for high-accuracy predictions
- **StandardScaler**: Feature normalization and standardization
- **GridSearchCV**: Hyperparameter tuning and model optimization

### Data Processing Libraries
- **DateTime**: Time-series data handling and timestamp management
- **SciPy**: Statistical analysis and advanced mathematical operations
- **Logging**: System monitoring and error tracking capabilities

### Visualization Dependencies
- **Plotly Express**: High-level plotting interface for quick visualizations
- **Plotly Graph Objects**: Low-level plotting control for custom interactive charts
- **Plotly Subplots**: Multi-panel dashboard creation and complex layout management