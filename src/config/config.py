"""
Configuration Module for Fraud Detection Project

This module centralizes all configuration settings and parameters
for the fraud detection project.
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionConfig:
    """
    Configuration class for fraud detection project.
    """
    
    def __init__(self):
        """Initialize configuration with default values."""
        
        # Project paths
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.EXTERNAL_DATA_DIR = self.DATA_DIR / "external"
        
        # Data file paths
        self.FRAUD_DATA_PATH = self.RAW_DATA_DIR / "Fraud_Data.csv"
        self.IP_COUNTRY_DATA_PATH = self.RAW_DATA_DIR / "IpAddress_to_Country.csv"
        self.CREDITCARD_DATA_PATH = self.RAW_DATA_DIR / "creditcard.csv"
        
        # Output paths
        self.MODELS_DIR = self.PROJECT_ROOT / "models"
        self.LOGS_DIR = self.PROJECT_ROOT / "logs"
        self.RESULTS_DIR = self.PROJECT_ROOT / "results"
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Data preprocessing settings
        self.PREPROCESSING_CONFIG = {
            'random_state': 42,
            'test_size': 0.2,
            'validation_size': 0.1,
            'missing_value_strategy': 'auto',  # 'auto', 'drop', 'impute'
            'scaling_method': 'standard',  # 'standard', 'minmax', 'robust'
            'categorical_encoding': 'label',  # 'label', 'onehot', 'target'
        }
        
        # Feature engineering settings
        self.FEATURE_ENGINEERING_CONFIG = {
            'time_features': True,
            'transaction_features': True,
            'location_features': True,
            'user_behavior_features': True,
            'interaction_features': True,
            'pca_features': True,
        }
        
        # Class imbalance handling
        self.CLASS_IMBALANCE_CONFIG = {
            'method': 'smote',  # 'smote', 'undersample', 'smoteenn', 'none'
            'sampling_strategy': 'auto',
            'k_neighbors': 5,
            'random_state': 42,
        }
        
        # Model training settings
        self.MODEL_CONFIG = {
            'random_state': 42,
            'cv_folds': 5,
            'n_jobs': -1,
            'verbose': 1,
        }
        
        # Model hyperparameters
        self.MODEL_HYPERPARAMS = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample'],
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'scale_pos_weight': [1, 5, 10],
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'class_weight': ['balanced'],
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'class_weight': ['balanced'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000],
            },
        }
        
        # Evaluation metrics
        self.EVALUATION_METRICS = [
            'accuracy',
            'precision',
            'recall',
            'f1',
            'roc_auc',
            'pr_auc',
            'specificity',
            'sensitivity',
        ]
        
        # Threshold optimization
        self.THRESHOLD_OPTIMIZATION = {
            'method': 'f1',  # 'f1', 'precision', 'recall', 'custom'
            'custom_weights': {'precision': 0.3, 'recall': 0.7},
            'threshold_range': (0.1, 0.9, 0.01),
        }
        
        # Feature selection
        self.FEATURE_SELECTION = {
            'method': 'correlation',  # 'correlation', 'mutual_info', 'recursive', 'lasso'
            'threshold': 0.01,
            'max_features': 50,
            'random_state': 42,
        }
        
        # Model explainability
        self.EXPLAINABILITY_CONFIG = {
            'use_shap': True,
            'shap_background_samples': 100,
            'feature_importance_method': 'permutation',  # 'permutation', 'shap', 'builtin'
            'top_features': 20,
        }
        
        # Logging configuration
        self.LOGGING_CONFIG = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': self.LOGS_DIR / 'fraud_detection.log',
        }
        
        # Database configuration (for future use)
        self.DATABASE_CONFIG = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', 5432),
            'database': os.getenv('DB_NAME', 'fraud_detection'),
            'username': os.getenv('DB_USER', 'fraud_user'),
            'password': os.getenv('DB_PASSWORD', 'fraud_password'),
        }
        
        # API configuration
        self.API_CONFIG = {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False,
            'reload': True,
        }
        
        # MLflow configuration
        self.MLFLOW_CONFIG = {
            'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
            'experiment_name': 'fraud_detection',
            'artifact_location': str(self.MODELS_DIR),
        }
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.EXTERNAL_DATA_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
            self.RESULTS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def get_data_paths(self) -> Dict[str, Path]:
        """Get all data file paths."""
        return {
            'fraud_data': self.FRAUD_DATA_PATH,
            'ip_country_data': self.IP_COUNTRY_DATA_PATH,
            'creditcard_data': self.CREDITCARD_DATA_PATH,
        }
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Get all output directory paths."""
        return {
            'models': self.MODELS_DIR,
            'logs': self.LOGS_DIR,
            'results': self.RESULTS_DIR,
            'processed_data': self.PROCESSED_DATA_DIR,
        }
    
    def update_config(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration settings."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
        }
        
        # Check if data files exist
        data_paths = self.get_data_paths()
        for name, path in data_paths.items():
            if not path.exists():
                validation_results['warnings'].append(f"Data file not found: {name} at {path}")
        
        # Validate preprocessing settings
        if self.PREPROCESSING_CONFIG['test_size'] + self.PREPROCESSING_CONFIG['validation_size'] >= 1.0:
            validation_results['errors'].append("Test size + validation size must be less than 1.0")
            validation_results['is_valid'] = False
        
        # Validate class imbalance method
        valid_imbalance_methods = ['smote', 'undersample', 'smoteenn', 'none']
        if self.CLASS_IMBALANCE_CONFIG['method'] not in valid_imbalance_methods:
            validation_results['errors'].append(f"Invalid class imbalance method: {self.CLASS_IMBALANCE_CONFIG['method']}")
            validation_results['is_valid'] = False
        
        # Validate scaling method
        valid_scaling_methods = ['standard', 'minmax', 'robust']
        if self.PREPROCESSING_CONFIG['scaling_method'] not in valid_scaling_methods:
            validation_results['errors'].append(f"Invalid scaling method: {self.PREPROCESSING_CONFIG['scaling_method']}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    def get_config_summary(self) -> str:
        """Get a summary of the current configuration."""
        summary = []
        summary.append("=" * 60)
        summary.append("FRAUD DETECTION PROJECT CONFIGURATION")
        summary.append("=" * 60)
        
        # Project paths
        summary.append("\n📁 PROJECT PATHS:")
        summary.append(f"  Project Root: {self.PROJECT_ROOT}")
        summary.append(f"  Data Directory: {self.DATA_DIR}")
        summary.append(f"  Models Directory: {self.MODELS_DIR}")
        summary.append(f"  Results Directory: {self.RESULTS_DIR}")
        
        # Data preprocessing
        summary.append("\n🔧 PREPROCESSING SETTINGS:")
        for key, value in self.PREPROCESSING_CONFIG.items():
            summary.append(f"  {key}: {value}")
        
        # Class imbalance
        summary.append("\n⚖️ CLASS IMBALANCE HANDLING:")
        for key, value in self.CLASS_IMBALANCE_CONFIG.items():
            summary.append(f"  {key}: {value}")
        
        # Model settings
        summary.append("\n🤖 MODEL SETTINGS:")
        for key, value in self.MODEL_CONFIG.items():
            summary.append(f"  {key}: {value}")
        
        # Evaluation
        summary.append("\n📊 EVALUATION METRICS:")
        summary.append(f"  Metrics: {', '.join(self.EVALUATION_METRICS)}")
        summary.append(f"  Threshold Optimization: {self.THRESHOLD_OPTIMIZATION['method']}")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)


# Global configuration instance
config = FraudDetectionConfig()


def get_config() -> FraudDetectionConfig:
    """Get the global configuration instance."""
    return config


def main():
    """Example usage of the configuration."""
    
    # Get configuration
    cfg = get_config()
    
    # Display configuration summary
    print(cfg.get_config_summary())
    
    # Validate configuration
    validation = cfg.validate_config()
    print(f"\nConfiguration validation: {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}")
    
    if validation['warnings']:
        print("\n⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"  • {warning}")
    
    if validation['errors']:
        print("\n❌ Errors:")
        for error in validation['errors']:
            print(f"  • {error}")


if __name__ == "__main__":
    main() 