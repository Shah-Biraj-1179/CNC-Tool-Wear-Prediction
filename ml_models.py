import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
import os

class ToolWearPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['vibration', 'temperature', 'cutting_force', 'spindle_speed', 'feed_rate', 'cutting_time']
        self.target_column = 'wear_level'
        self.is_trained = False
        self.model_type = None
        self.metrics = {}
        
    def prepare_features(self, data):
        """Prepare features for training/prediction"""
        try:
            # Create cutting_time if not present
            if 'cutting_time' not in data.columns:
                data = data.copy()
                data['cutting_time'] = np.random.uniform(1, 50, len(data))  # Placeholder
            
            # Select and validate features
            available_features = [col for col in self.feature_columns if col in data.columns]
            
            if not available_features:
                raise ValueError("No valid feature columns found in data")
            
            X = data[available_features].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Feature engineering
            X['vibration_temp_ratio'] = X['vibration'] / (X['temperature'] + 1)
            X['force_speed_ratio'] = X['cutting_force'] / (X['spindle_speed'] + 1)
            X['power_estimate'] = X['cutting_force'] * X['feed_rate'] * X['spindle_speed'] / 1000
            
            return X
        
        except Exception as e:
            raise Exception(f"Feature preparation failed: {str(e)}")
    
    def train_model(self, data, model_type='random_forest', test_size=0.2, cv=True, **params):
        """Train the machine learning model"""
        try:
            # Prepare features and target
            X = self.prepare_features(data)
            y = data[self.target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=pd.cut(y, bins=3, labels=[0,1,2])
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize model based on type
            if model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    random_state=42,
                    n_jobs=-1
                )
                X_train_final = X_train  # Random Forest doesn't need scaling
                X_test_final = X_test
            
            elif model_type == 'support_vector_machine':
                self.model = SVR(
                    kernel=params.get('kernel', 'rbf'),
                    gamma='scale',
                    C=1.0
                )
                X_train_final = X_train_scaled
                X_test_final = X_test_scaled
            
            elif model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(
                    learning_rate=params.get('learning_rate', 0.1),
                    n_estimators=100,
                    random_state=42
                )
                X_train_final = X_train
                X_test_final = X_test
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            self.model.fit(X_train_final, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_final)
            
            # Convert to classification for some metrics
            y_test_class = (y_test.values > 0.7).astype(int)
            y_pred_class = (y_pred > 0.7).astype(int)
            
            # Calculate metrics
            results = {
                'accuracy': accuracy_score(y_test_class, y_pred_class),
                'precision': precision_score(y_test_class, y_pred_class, average='weighted', zero_division='warn'),
                'recall': recall_score(y_test_class, y_pred_class, average='weighted', zero_division='warn'),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test_class, y_pred_class)
            }
            
            # Feature importance (for tree-based models)
            if hasattr(self.model, 'feature_importances_'):
                results['feature_importance'] = self.model.feature_importances_
                results['features'] = list(X_train.columns)
            
            # Cross-validation
            if cv:
                cv_scores = cross_val_score(self.model, X_train_final, y_train, cv=5, scoring='r2')
                results['cv_score_mean'] = cv_scores.mean()
                results['cv_score_std'] = cv_scores.std()
            
            self.metrics = results
            self.model_type = model_type
            self.is_trained = True
            
            return results
            
        except Exception as e:
            raise Exception(f"Model training failed: {str(e)}")
    
    def predict(self, data):
        """Make predictions on new data"""
        if not self.is_trained:
            raise Exception("Model not trained yet")
        
        try:
            X = self.prepare_features(data)
            
            # Apply same preprocessing as training
            if self.model_type == 'support_vector_machine':
                X_scaled = self.scaler.transform(X)
                predictions = self.model.predict(X_scaled)
            else:
                predictions = self.model.predict(X)
            
            return predictions
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def predict_proba(self, data):
        """Get prediction probabilities (approximation for regression)"""
        if not self.is_trained:
            return None
        
        try:
            predictions = self.predict(data)
            
            # Convert regression output to pseudo-probabilities
            # This is a simplified approach for demonstration
            proba = np.column_stack([
                1 - predictions,  # Low wear probability
                predictions       # High wear probability
            ])
            
            # Normalize to ensure probabilities sum to 1
            proba = proba / proba.sum(axis=1, keepdims=True)
            
            return proba
            
        except Exception:
            return None
    
    def get_model_accuracy(self):
        """Get model accuracy"""
        if not self.is_trained or not self.metrics:
            return None
        return self.metrics.get('accuracy')
    
    def get_detailed_metrics(self):
        """Get detailed model metrics"""
        return self.metrics if self.is_trained else {}
    
    def save_model(self, filepath):
        """Save trained model"""
        if not self.is_trained:
            raise Exception("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load saved model"""
        if not os.path.exists(filepath):
            raise Exception("Model file not found")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.metrics = model_data['metrics']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True
