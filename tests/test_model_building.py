"""
Tests for model building and training functionality.

This module contains unit tests for the fraud detection model building pipeline.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for importing our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.model_builder import FraudModelBuilder
from utils.model_evaluation import FraudModelEvaluator
from config.config import get_config


class TestFraudModelBuilder(unittest.TestCase):
    """Test cases for FraudModelBuilder class."""
    
    def setUp(self):
        """Set up test data."""
        self.model_builder = FraudModelBuilder(random_state=42)
        
        # Create sample data for testing
        np.random.seed(42)
        n_samples = 1000
        
        # Create features
        feature_data = {
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            'feature_4': np.random.normal(0, 1, n_samples),
            'feature_5': np.random.normal(0, 1, n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'purchase_value': np.random.exponential(100, n_samples),
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples),
            'is_night': np.random.choice([0, 1], n_samples),
            'log_purchase_value': np.log1p(np.random.exponential(100, n_samples)),
            'is_high_value': np.random.choice([0, 1], n_samples),
            'is_low_value': np.random.choice([0, 1], n_samples),
            'user_transaction_count': np.random.randint(1, 10, n_samples),
            'is_first_time_user': np.random.choice([0, 1], n_samples),
            'device_transaction_count': np.random.randint(1, 5, n_samples),
            'gender_encoded': np.random.choice([0, 1], n_samples),
            'device_encoded': np.random.randint(1, 4, n_samples),
            'browser_encoded': np.random.randint(1, 5, n_samples),
            'source_encoded': np.random.randint(1, 5, n_samples),
            'class': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% fraud rate
        }
        
        self.sample_data = pd.DataFrame(feature_data)
    
    def test_prepare_data(self):
        """Test data preparation functionality."""
        # Test data preparation
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.model_builder.prepare_data(
            self.sample_data, target_column='class', test_size=0.2, validation_size=0.1
        )
        
        # Check that data is properly split
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_val), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_val), len(y_val))
        self.assertEqual(len(X_test), len(y_test))
        
        # Check that target column is excluded from features
        self.assertNotIn('class', feature_names)
        
        # Check that non-feature columns are excluded
        exclude_columns = ['user_id', 'signup_time', 'purchase_time', 'ip_address']
        for col in exclude_columns:
            self.assertNotIn(col, feature_names)
    
    def test_scale_features(self):
        """Test feature scaling functionality."""
        # Prepare data first
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.model_builder.prepare_data(
            self.sample_data, target_column='class', test_size=0.2, validation_size=0.1
        )
        
        # Test scaling
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = self.model_builder.scale_features(
            X_train, X_val, X_test, method='standard'
        )
        
        # Check that scaling was applied
        self.assertAlmostEqual(X_train_scaled.mean().mean(), 0, places=1)
        self.assertAlmostEqual(X_train_scaled.std().mean(), 1, places=1)
        
        # Check that scaler is stored
        self.assertIn('standard', self.model_builder.scalers)
    
    def test_build_logistic_regression(self):
        """Test Logistic Regression model building."""
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.model_builder.prepare_data(
            self.sample_data, target_column='class', test_size=0.2, validation_size=0.1
        )
        
        # Build model without hyperparameter tuning for speed
        lr_model = self.model_builder.build_logistic_regression(
            X_train, y_train, X_val, y_val, hyperparameter_tuning=False
        )
        
        # Check that model is built
        self.assertIsNotNone(lr_model)
        self.assertIn('logistic_regression', self.model_builder.models)
        
        # Check that model can make predictions
        predictions = lr_model.predict(X_test)
        self.assertEqual(len(predictions), len(y_test))
    
    def test_build_random_forest(self):
        """Test Random Forest model building."""
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.model_builder.prepare_data(
            self.sample_data, target_column='class', test_size=0.2, validation_size=0.1
        )
        
        # Build model without hyperparameter tuning for speed
        rf_model = self.model_builder.build_random_forest(
            X_train, y_train, X_val, y_val, hyperparameter_tuning=False
        )
        
        # Check that model is built
        self.assertIsNotNone(rf_model)
        self.assertIn('random_forest', self.model_builder.models)
        
        # Check that feature importance is available
        self.assertIn('random_forest', self.model_builder.feature_importance)
        
        # Check that model can make predictions
        predictions = rf_model.predict(X_test)
        self.assertEqual(len(predictions), len(y_test))
    
    def test_evaluate_models(self):
        """Test model evaluation functionality."""
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.model_builder.prepare_data(
            self.sample_data, target_column='class', test_size=0.2, validation_size=0.1
        )
        
        # Build a simple model
        lr_model = self.model_builder.build_logistic_regression(
            X_train, y_train, X_val, y_val, hyperparameter_tuning=False
        )
        
        # Evaluate model
        evaluation_results = self.model_builder.evaluate_models(X_test, y_test)
        
        # Check that evaluation results are returned
        self.assertIn('logistic_regression', evaluation_results)
        self.assertIn('metrics', evaluation_results['logistic_regression'])
        self.assertIn('confusion_matrix', evaluation_results['logistic_regression'])
    
    def test_select_best_model(self):
        """Test best model selection."""
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.model_builder.prepare_data(
            self.sample_data, target_column='class', test_size=0.2, validation_size=0.1
        )
        
        # Build multiple models
        lr_model = self.model_builder.build_logistic_regression(
            X_train, y_train, X_val, y_val, hyperparameter_tuning=False
        )
        rf_model = self.model_builder.build_random_forest(
            X_train, y_train, X_val, y_val, hyperparameter_tuning=False
        )
        
        # Evaluate models
        self.model_builder.evaluate_models(X_test, y_test)
        
        # Select best model
        best_model_name = self.model_builder.select_best_model(metric='f1_score')
        
        # Check that best model is selected
        self.assertIsNotNone(best_model_name)
        self.assertIn(best_model_name, self.model_builder.models)
        self.assertEqual(self.model_builder.best_model_name, best_model_name)
    
    def test_get_model_summary(self):
        """Test model summary generation."""
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.model_builder.prepare_data(
            self.sample_data, target_column='class', test_size=0.2, validation_size=0.1
        )
        
        # Build a model
        lr_model = self.model_builder.build_logistic_regression(
            X_train, y_train, X_val, y_val, hyperparameter_tuning=False
        )
        
        # Get summary
        summary = self.model_builder.get_model_summary()
        
        # Check that summary is generated
        self.assertIn('models_trained', summary)
        self.assertIn('feature_importance_available', summary)
        self.assertIsInstance(summary['models_trained'], list)


class TestFraudModelEvaluator(unittest.TestCase):
    """Test cases for FraudModelEvaluator class."""
    
    def setUp(self):
        """Set up test data."""
        self.evaluator = FraudModelEvaluator()
        
        # Create sample predictions
        np.random.seed(42)
        n_samples = 1000
        
        self.y_true = pd.Series(np.random.choice([0, 1], n_samples, p=[0.95, 0.05]))
        self.y_pred = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        self.y_pred_proba = np.random.random(n_samples)
    
    def test_calculate_fraud_metrics(self):
        """Test fraud metrics calculation."""
        metrics = self.evaluator.calculate_fraud_metrics(self.y_true, self.y_pred, self.y_pred_proba)
        
        # Check that all required metrics are calculated
        required_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc',
            'specificity', 'sensitivity', 'false_positive_rate', 'false_negative_rate'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_evaluate_model_performance(self):
        """Test model performance evaluation."""
        # Create a mock model
        class MockModel:
            def predict(self, X):
                return np.random.choice([0, 1], len(X))
            
            def predict_proba(self, X):
                return np.column_stack([np.random.random(len(X)), np.random.random(len(X))])
        
        mock_model = MockModel()
        
        # Create test data
        X_test = pd.DataFrame(np.random.random((100, 5)))
        y_test = pd.Series(np.random.choice([0, 1], 100, p=[0.95, 0.05]))
        
        # Evaluate model
        results = self.evaluator.evaluate_model_performance(mock_model, X_test, y_test, "MockModel")
        
        # Check that results are returned
        self.assertIn('MockModel', self.evaluator.evaluation_results)
        self.assertIn('metrics', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('predictions', results)
        self.assertIn('probabilities', results)
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Create mock models
        class MockModel:
            def predict(self, X):
                return np.random.choice([0, 1], len(X))
            
            def predict_proba(self, X):
                return np.column_stack([np.random.random(len(X)), np.random.random(len(X))])
        
        models = {
            'Model1': MockModel(),
            'Model2': MockModel()
        }
        
        # Create test data
        X_test = pd.DataFrame(np.random.random((100, 5)))
        y_test = pd.Series(np.random.choice([0, 1], 100, p=[0.95, 0.05]))
        
        # Compare models
        comparison_df = self.evaluator.compare_models(models, X_test, y_test)
        
        # Check that comparison table is generated
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertEqual(len(comparison_df), 2)  # Two models
        self.assertIn('Model', comparison_df.columns)
        self.assertIn('F1-Score', comparison_df.columns)
    
    def test_analyze_business_impact(self):
        """Test business impact analysis."""
        # First evaluate a model
        class MockModel:
            def predict(self, X):
                return np.random.choice([0, 1], len(X))
            
            def predict_proba(self, X):
                return np.column_stack([np.random.random(len(X)), np.random.random(len(X))])
        
        mock_model = MockModel()
        X_test = pd.DataFrame(np.random.random((100, 5)))
        y_test = pd.Series(np.random.choice([0, 1], 100, p=[0.95, 0.05]))
        
        self.evaluator.evaluate_model_performance(mock_model, X_test, y_test, "MockModel")
        
        # Analyze business impact
        business_impact = self.evaluator.analyze_business_impact("MockModel")
        
        # Check that business metrics are calculated
        required_business_metrics = [
            'total_transactions', 'fraud_rate', 'fraud_detection_rate',
            'false_alarm_rate', 'net_savings', 'roi_percentage'
        ]
        
        for metric in required_business_metrics:
            self.assertIn(metric, business_impact)
            self.assertIsInstance(business_impact[metric], (int, float))
    
    def test_get_best_model(self):
        """Test best model selection."""
        # Create evaluation results
        self.evaluator.evaluation_results = {
            'Model1': {'metrics': {'f1_score': 0.8}},
            'Model2': {'metrics': {'f1_score': 0.9}},
            'Model3': {'metrics': {'f1_score': 0.7}}
        }
        
        # Get best model
        best_model = self.evaluator.get_best_model(metric='f1_score')
        
        # Check that best model is selected
        self.assertEqual(best_model, 'Model2')
    
    def test_generate_evaluation_report(self):
        """Test evaluation report generation."""
        # Create evaluation results
        self.evaluator.evaluation_results = {
            'Model1': {
                'metrics': {
                    'accuracy': 0.95,
                    'precision': 0.8,
                    'recall': 0.7,
                    'f1_score': 0.75
                },
                'confusion_matrix': np.array([[90, 5], [3, 2]])
            }
        }
        
        # Generate report
        report = self.evaluator.generate_evaluation_report()
        
        # Check that report is generated
        self.assertIsInstance(report, str)
        self.assertIn('FRAUD DETECTION MODEL EVALUATION REPORT', report)
        self.assertIn('Model1', report)


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration management."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = get_config()
        
        # Check that configuration is created
        self.assertIsNotNone(config)
        self.assertIn('PREPROCESSING_CONFIG', config.__dict__)
        self.assertIn('MODEL_CONFIG', config.__dict__)
        self.assertIn('EVALUATION_METRICS', config.__dict__)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = get_config()
        validation = config.validate_config()
        
        # Check that validation results are returned
        self.assertIn('is_valid', validation)
        self.assertIn('errors', validation)
        self.assertIn('warnings', validation)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestFraudModelBuilder))
    test_suite.addTest(unittest.makeSuite(TestFraudModelEvaluator))
    test_suite.addTest(unittest.makeSuite(TestConfiguration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running fraud detection model building tests...")
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    print("\nTest execution completed.") 