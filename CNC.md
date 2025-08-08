# CNC Tool Wear Prediction System

## Overview

This is a comprehensive AI-powered CNC Tool Wear Prediction System built with Streamlit that provides predictive maintenance capabilities for industrial CNC operations. The system uses machine learning models to predict tool wear levels, schedules maintenance activities, and provides interactive visualizations to monitor tool health and performance. It's designed to help manufacturing facilities optimize their maintenance schedules, reduce downtime, and extend tool life through data-driven insights.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Framework
- **Frontend**: Streamlit-based web application with multi-page navigation
- **Architecture Pattern**: Modular design with separation of concerns across models, utilities, and data layers
- **Session Management**: Streamlit session state for maintaining application state across user interactions

### Machine Learning Pipeline
- **Model Architecture**: Ensemble approach supporting Random Forest, Gradient Boosting, and SVM regressors
- **Feature Engineering**: Automated feature creation including ratios and derived metrics (vibration-temperature ratio, force-speed ratio, power estimates)
- **Data Processing**: Comprehensive validation, cleaning, and preprocessing pipeline with outlier detection
- **Model Persistence**: Joblib-based model serialization for deployment

### Data Management
- **Sample Data Generation**: Built-in synthetic data generator with realistic CNC operational parameters
- **Data Validation**: Schema validation with required/optional column checking and data type enforcement
- **Feature Set**: Core sensors (vibration, temperature, cutting force, spindle speed, feed rate) plus derived metrics

### Predictive Maintenance System
- **Scheduling Engine**: Multi-tier maintenance scheduling (preventive, corrective, emergency) with cost optimization
- **Threshold Management**: Configurable wear level thresholds for different priority levels
- **Wear Progression**: Time-series based wear rate calculation for predictive scheduling

### Visualization Layer
- **Charting Engine**: Plotly-based interactive visualizations with consistent color schemes
- **Dashboard Components**: Trend analysis, wear level monitoring, and maintenance schedule views
- **Real-time Updates**: Dynamic chart updates based on model predictions and data changes

## External Dependencies

### Core Python Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and preprocessing tools
- **Plotly**: Interactive visualization and charting library
- **Joblib**: Model serialization and persistence

### Machine Learning Stack
- **Random Forest**: Primary ensemble method for wear prediction
- **Gradient Boosting**: Alternative ensemble approach for comparison
- **Support Vector Machines**: Non-linear regression capability
- **StandardScaler**: Feature normalization and preprocessing

### Data Processing Tools
- **DateTime utilities**: Timestamp handling and time-series operations
- **Logging**: Application monitoring and error tracking
- **Statistical libraries**: Data validation and outlier detection algorithms