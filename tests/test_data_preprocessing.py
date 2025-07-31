"""
Tests for data preprocessing functionality.

This module contains unit tests for the fraud detection data preprocessing pipeline.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for importing our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing.data_processor import FraudDataProcessor
from features.feature_engineering import FraudFeatureEngineer
from utils.data_validation import DataValidator
from config.config import FraudDetectionConfig


class TestFraudDataProcessor(unittest.TestCase):
    """Test cases for FraudDataProcessor class."""
    
    def setUp(self):
        """Set up test data."""
        self.processor = FraudDataProcessor(random_state=42)
        
        # Create sample fraud data
        np.random.seed(42)
        n_samples = 1000
        
        self.sample_fraud_data = pd.DataFrame({
            'user_id': range(1, n_samples + 1),
            'signup_time': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
            'purchase_time': pd.date_range('2024-01-01', periods=n_samples, freq='H') + 
                           pd.Timedelta(hours=np.random.randint(1, 168, n_samples)),
            'purchase_value': np.random.exponential(100, n_samples),
            'device_id': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
            'browser': np.random.choice(['chrome', 'firefox', 'safari'], n_samples),
            'source': np.random.choice(['direct', 'organic', 'paid'], n_samples),
            'ip_address': [f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" 
                          for _ in range(n_samples)],
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'class': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        })
        
        # Create sample IP country data
        self.sample_ip_country_data = pd.DataFrame({
            'ip_address': [f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" 
                          for _ in range(500)],
            'country': np.random.choice(['US', 'UK', 'CA', 'AU'], 500)
        })
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Add some missing values
        df_with_missing = self.sample_fraud_data.copy()
        df_with_missing.loc[0:10, 'purchase_value'] = np.nan
        df_with_missing.loc[0:5, 'gender'] = np.nan
        
        # Test automatic strategy
        df_clean = self.processor.handle_missing_values(df_with_missing, strategy='auto')
        
        # Check that missing values are handled
        self.assertEqual(df_clean.isnull().sum().sum(), 0)
        self.assertEqual(len(df_clean), len(df_with_missing))
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Add some duplicates
        df_with_duplicates = pd.concat([self.sample_fraud_data, 
                                      self.sample_fraud_data.iloc[0:10]])
        
        # Clean the data
        df_clean = self.processor.clean_data(df_with_duplicates)
        
        # Check that duplicates are removed
        self.assertEqual(len(df_clean), len(self.sample_fraud_data))
        
        # Check that IP addresses are converted to integers
        if 'ip_address_int' in df_clean.columns:
            self.assertTrue(df_clean['ip_address_int'].dtype in ['int64', 'int32'])
    
    def test_create_time_features(self):
        """Test time-based feature creation."""
        df_with_time_features = self.processor.create_time_features(self.sample_fraud_data)
        
        # Check that time features are created
        expected_features = ['hour_of_day', 'day_of_week', 'month', 'day_of_month']
        for feature in expected_features:
            self.assertIn(feature, df_with_time_features.columns)
        
        # Check that hour_of_day is in valid range
        self.assertTrue(all(0 <= df_with_time_features['hour_of_day'] <= 23))
    
    def test_create_transaction_features(self):
        """Test transaction feature creation."""
        df_with_transaction_features = self.processor.create_transaction_features(self.sample_fraud_data)
        
        # Check that transaction features are created
        expected_features = ['log_purchase_value', 'is_high_value', 'is_low_value']
        for feature in expected_features:
            self.assertIn(feature, df_with_transaction_features.columns)
        
        # Check that log transformation is applied
        self.assertTrue(all(df_with_transaction_features['log_purchase_value'] >= 0))
    
    def test_merge_datasets(self):
        """Test dataset merging functionality."""
        merged_df = self.processor.merge_datasets(self.sample_fraud_data, self.sample_ip_country_data)
        
        # Check that merge was successful
        self.assertIn('country', merged_df.columns)
        self.assertEqual(len(merged_df), len(self.sample_fraud_data))
    
    def test_handle_class_imbalance(self):
        """Test class imbalance handling."""
        # Prepare features and target
        df_features = self.processor.create_time_features(self.sample_fraud_data)
        df_features = self.processor.create_transaction_features(df_features)
        
        # Select numerical features
        numerical_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features = [col for col in numerical_features if col != 'class']
        
        X = df_features[numerical_features]
        y = df_features['class']
        
        # Test SMOTE
        X_train_resampled, y_train_resampled, X_test, y_test = self.processor.handle_class_imbalance(
            X, y, method='smote', test_size=0.2
        )
        
        # Check that resampling was successful
        self.assertGreater(len(X_train_resampled), len(X) * 0.7)  # SMOTE increases size
        self.assertEqual(len(y_train_resampled), len(X_train_resampled))
    
    def test_scale_features(self):
        """Test feature scaling."""
        # Prepare features
        df_features = self.processor.create_time_features(self.sample_fraud_data)
        df_features = self.processor.create_transaction_features(df_features)
        
        numerical_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features = [col for col in numerical_features if col != 'class']
        
        X = df_features[numerical_features]
        y = df_features['class']
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.processor.scale_features(
            X_train, X_test, method='standard'
        )
        
        # Check that scaling was applied
        self.assertAlmostEqual(X_train_scaled.mean().mean(), 0, places=1)
        self.assertAlmostEqual(X_train_scaled.std().mean(), 1, places=1)


class TestFraudFeatureEngineer(unittest.TestCase):
    """Test cases for FraudFeatureEngineer class."""
    
    def setUp(self):
        """Set up test data."""
        self.engineer = FraudFeatureEngineer()
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        
        self.sample_data = pd.DataFrame({
            'user_id': range(1, n_samples + 1),
            'signup_time': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
            'purchase_time': pd.date_range('2024-01-01', periods=n_samples, freq='H') + 
                           pd.Timedelta(hours=np.random.randint(1, 168, n_samples)),
            'purchase_value': np.random.exponential(100, n_samples),
            'device_id': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
            'browser': np.random.choice(['chrome', 'firefox', 'safari'], n_samples),
            'source': np.random.choice(['direct', 'organic', 'paid'], n_samples),
            'ip_address': [f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" 
                          for _ in range(n_samples)],
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'class': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        })
    
    def test_create_ecommerce_features(self):
        """Test e-commerce feature creation."""
        features_df = self.engineer.create_ecommerce_features(self.sample_data)
        
        # Check that features are created
        expected_features = [
            'signup_to_purchase_hours', 'purchase_hour', 'day_of_week',
            'is_night_purchase', 'is_weekend_purchase', 'quick_purchase_flag'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features_df.columns)
    
    def test_get_feature_importance_ranking(self):
        """Test feature importance ranking."""
        features_df = self.engineer.create_ecommerce_features(self.sample_data)
        
        ranking = self.engineer.get_feature_importance_ranking(features_df, target_col='class')
        
        # Check that ranking is created
        self.assertIsInstance(ranking, pd.DataFrame)
        self.assertGreater(len(ranking), 0)
        self.assertIn('feature', ranking.columns)
        self.assertIn('correlation', ranking.columns)
    
    def test_select_top_features(self):
        """Test top feature selection."""
        features_df = self.engineer.create_ecommerce_features(self.sample_data)
        
        top_features = self.engineer.select_top_features(features_df, target_col='class', top_n=10)
        
        # Check that top features are selected
        self.assertIsInstance(top_features, list)
        self.assertLessEqual(len(top_features), 10)


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""
    
    def setUp(self):
        """Set up test data."""
        self.validator = DataValidator()
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        self.sample_fraud_data = pd.DataFrame({
            'user_id': range(1, n_samples + 1),
            'purchase_time': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
            'purchase_value': np.random.exponential(100, n_samples),
            'class': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        })
    
    def test_validate_fraud_data(self):
        """Test fraud data validation."""
        results = self.validator.validate_fraud_data(self.sample_fraud_data)
        
        # Check that validation results are returned
        self.assertIn('is_valid', results)
        self.assertIn('errors', results)
        self.assertIn('warnings', results)
        self.assertIn('data_quality_score', results)
        
        # Check that data is valid
        self.assertTrue(results['is_valid'])
        self.assertGreater(results['data_quality_score'], 0.5)
    
    def test_generate_validation_report(self):
        """Test validation report generation."""
        self.validator.validate_fraud_data(self.sample_fraud_data)
        report = self.validator.generate_validation_report()
        
        # Check that report is generated
        self.assertIsInstance(report, str)
        self.assertIn('DATA VALIDATION REPORT', report)


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration management."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = FraudDetectionConfig()
        
        # Check that configuration is created
        self.assertIsInstance(config.PREPROCESSING_CONFIG, dict)
        self.assertIsInstance(config.MODEL_CONFIG, dict)
        self.assertIsInstance(config.EVALUATION_METRICS, list)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = FraudDetectionConfig()
        validation = config.validate_config()
        
        # Check that validation results are returned
        self.assertIn('is_valid', validation)
        self.assertIn('errors', validation)
        self.assertIn('warnings', validation)
    
    def test_config_summary(self):
        """Test configuration summary generation."""
        config = FraudDetectionConfig()
        summary = config.get_config_summary()
        
        # Check that summary is generated
        self.assertIsInstance(summary, str)
        self.assertIn('FRAUD DETECTION PROJECT CONFIGURATION', summary)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestFraudDataProcessor))
    test_suite.addTest(unittest.makeSuite(TestFraudFeatureEngineer))
    test_suite.addTest(unittest.makeSuite(TestDataValidator))
    test_suite.addTest(unittest.makeSuite(TestConfiguration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running fraud detection preprocessing tests...")
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    print("\nTest execution completed.") 