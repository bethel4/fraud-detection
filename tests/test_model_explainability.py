"""
Unit tests for model explainability functionality
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.model_explainability import FraudModelExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

class TestFraudModelExplainer(unittest.TestCase):
    """Test cases for FraudModelExplainer class"""
    
    def setUp(self):
        """Set up test data and models"""
        # Create sample data
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 10
        
        self.X = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, self.n_samples),
            'feature_2': np.random.exponential(1, self.n_samples),
            'feature_3': np.random.uniform(0, 10, self.n_samples),
            'feature_4': np.random.choice([0, 1], self.n_samples),
            'feature_5': np.random.normal(5, 2, self.n_samples),
            'feature_6': np.random.poisson(3, self.n_samples),
            'feature_7': np.random.beta(2, 5, self.n_samples),
            'feature_8': np.random.lognormal(0, 1, self.n_samples),
            'feature_9': np.random.gamma(2, 2, self.n_samples),
            'feature_10': np.random.weibull(2, self.n_samples)
        })
        
        # Create target variable
        self.y = np.random.choice([0, 1], self.n_samples, p=[0.8, 0.2])
        
        # Train different types of models
        self.rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.rf_model.fit(self.X, self.y)
        
        self.lr_model = LogisticRegression(random_state=42)
        self.lr_model.fit(self.X, self.y)
        
        # Create temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    def test_init_random_forest(self):
        """Test initialization with Random Forest model"""
        explainer = FraudModelExplainer(self.rf_model)
        self.assertEqual(explainer.model_type, 'random_forest')
        self.assertIsNotNone(explainer.model)
        self.assertEqual(explainer.random_state, 42)
    
    def test_init_logistic_regression(self):
        """Test initialization with Logistic Regression model"""
        explainer = FraudModelExplainer(self.lr_model)
        self.assertEqual(explainer.model_type, 'logistic')
        self.assertIsNotNone(explainer.model)
    
    def test_detect_model_type(self):
        """Test automatic model type detection"""
        explainer = FraudModelExplainer(self.rf_model)
        
        # Test different model types
        self.assertEqual(explainer._detect_model_type(), 'random_forest')
        
        # Test with explicit model type
        explainer = FraudModelExplainer(self.rf_model, model_type='xgboost')
        self.assertEqual(explainer.model_type, 'xgboost')
    
    def test_explain_model_random_forest(self):
        """Test SHAP explanation generation for Random Forest"""
        explainer = FraudModelExplainer(self.rf_model)
        results = explainer.explain_model(self.X, sample_size=50)
        
        self.assertIn('shap_values', results)
        self.assertIn('expected_value', results)
        self.assertIn('feature_names', results)
        self.assertIn('sample_data', results)
        
        self.assertEqual(len(results['shap_values']), 50)
        self.assertEqual(len(results['feature_names']), self.n_features)
        self.assertEqual(results['sample_data'].shape, (50, self.n_features))
    
    def test_explain_model_logistic_regression(self):
        """Test SHAP explanation generation for Logistic Regression"""
        explainer = FraudModelExplainer(self.lr_model)
        results = explainer.explain_model(self.X, sample_size=30)
        
        self.assertIn('shap_values', results)
        self.assertIn('expected_value', results)
        self.assertIn('feature_names', results)
        self.assertIn('sample_data', results)
        
        self.assertEqual(len(results['shap_values']), 30)
    
    def test_get_feature_importance_ranking(self):
        """Test feature importance ranking"""
        explainer = FraudModelExplainer(self.rf_model)
        explainer.explain_model(self.X, sample_size=50)
        
        importance_df = explainer.get_feature_importance_ranking(top_n=5)
        
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertEqual(len(importance_df), 5)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('mean_abs_shap', importance_df.columns)
        self.assertIn('std_shap', importance_df.columns)
        
        # Check sorting
        importance_values = importance_df['mean_abs_shap'].values
        self.assertTrue(np.all(importance_values[:-1] >= importance_values[1:]))
    
    def test_analyze_fraud_drivers(self):
        """Test fraud drivers analysis"""
        explainer = FraudModelExplainer(self.rf_model)
        explainer.explain_model(self.X, sample_size=50)
        
        fraud_drivers = explainer.analyze_fraud_drivers(top_features=5)
        
        self.assertIsInstance(fraud_drivers, dict)
        self.assertEqual(len(fraud_drivers), 5)
        
        for feature, analysis in fraud_drivers.items():
            self.assertIn('importance', analysis)
            self.assertIn('std_importance', analysis)
            self.assertIn('positive_effect', analysis)
            self.assertIn('negative_effect', analysis)
            self.assertIn('mean_value', analysis)
            self.assertIn('std_value', analysis)
            self.assertIn('effect_direction', analysis)
            
            self.assertIn(analysis['effect_direction'], ['positive', 'negative'])
    
    def test_create_summary_plot(self):
        """Test summary plot creation"""
        explainer = FraudModelExplainer(self.rf_model)
        explainer.explain_model(self.X, sample_size=50)
        
        # Test without saving
        fig = explainer.create_summary_plot(show=False)
        self.assertIsNotNone(fig)
        
        # Test with saving
        save_path = os.path.join(self.temp_dir, 'summary_plot.png')
        fig = explainer.create_summary_plot(save_path=save_path, show=False)
        self.assertTrue(os.path.exists(save_path))
    
    def test_create_force_plot(self):
        """Test force plot creation"""
        explainer = FraudModelExplainer(self.rf_model)
        explainer.explain_model(self.X, sample_size=50)
        
        # Test without saving
        fig = explainer.create_force_plot(instance_idx=0, show=False)
        self.assertIsNotNone(fig)
        
        # Test with saving
        save_path = os.path.join(self.temp_dir, 'force_plot.png')
        fig = explainer.create_force_plot(instance_idx=0, save_path=save_path, show=False)
        self.assertTrue(os.path.exists(save_path))
    
    def test_create_waterfall_plot(self):
        """Test waterfall plot creation"""
        explainer = FraudModelExplainer(self.rf_model)
        explainer.explain_model(self.X, sample_size=50)
        
        # Test without saving
        fig = explainer.create_waterfall_plot(instance_idx=0, show=False)
        self.assertIsNotNone(fig)
        
        # Test with saving
        save_path = os.path.join(self.temp_dir, 'waterfall_plot.png')
        fig = explainer.create_waterfall_plot(instance_idx=0, save_path=save_path, show=False)
        self.assertTrue(os.path.exists(save_path))
    
    def test_create_dependence_plot(self):
        """Test dependence plot creation"""
        explainer = FraudModelExplainer(self.rf_model)
        explainer.explain_model(self.X, sample_size=50)
        
        # Test without saving
        fig = explainer.create_dependence_plot('feature_1', show=False)
        self.assertIsNotNone(fig)
        
        # Test with saving
        save_path = os.path.join(self.temp_dir, 'dependence_plot.png')
        fig = explainer.create_dependence_plot('feature_1', save_path=save_path, show=False)
        self.assertTrue(os.path.exists(save_path))
    
    def test_create_comprehensive_report(self):
        """Test comprehensive report generation"""
        explainer = FraudModelExplainer(self.rf_model)
        explainer.explain_model(self.X, sample_size=50)
        
        output_dir = os.path.join(self.temp_dir, 'explainability_report')
        report = explainer.create_comprehensive_report(output_dir)
        
        self.assertIsInstance(report, dict)
        self.assertIn('plots_generated', report)
        self.assertIn('output_directory', report)
        self.assertIn('fraud_drivers', report)
        self.assertIn('feature_importance', report)
        self.assertIn('model_type', report)
        self.assertIn('samples_analyzed', report)
        
        # Check that files were created
        self.assertTrue(os.path.exists(output_dir))
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'explainability_report.txt')))
        
        # Check that some plots were generated
        self.assertGreater(report['plots_generated'], 0)
    
    def test_interpret_fraud_patterns(self):
        """Test fraud pattern interpretation"""
        explainer = FraudModelExplainer(self.rf_model)
        explainer.explain_model(self.X, sample_size=50)
        
        insights = explainer.interpret_fraud_patterns()
        
        self.assertIsInstance(insights, dict)
        self.assertIn('high_risk_features', insights)
        self.assertIn('protective_features', insights)
        self.assertIn('behavioral_patterns', insights)
        self.assertIn('recommendations', insights)
        
        # Check that insights are lists
        self.assertIsInstance(insights['high_risk_features'], list)
        self.assertIsInstance(insights['protective_features'], list)
        self.assertIsInstance(insights['recommendations'], list)
    
    def test_error_handling_no_shap_values(self):
        """Test error handling when SHAP values not generated"""
        explainer = FraudModelExplainer(self.rf_model)
        
        # Try to use methods without generating SHAP values
        with self.assertRaises(ValueError):
            explainer.get_feature_importance_ranking()
        
        with self.assertRaises(ValueError):
            explainer.analyze_fraud_drivers()
        
        with self.assertRaises(ValueError):
            explainer.create_summary_plot()
    
    def test_sample_size_handling(self):
        """Test handling of different sample sizes"""
        explainer = FraudModelExplainer(self.rf_model)
        
        # Test with sample size larger than data
        results = explainer.explain_model(self.X, sample_size=200)
        self.assertEqual(len(results['shap_values']), self.n_samples)
        
        # Test with no sample size (use all data)
        results = explainer.explain_model(self.X)
        self.assertEqual(len(results['shap_values']), self.n_samples)
    
    def test_model_compatibility(self):
        """Test compatibility with different model types"""
        # Test XGBoost model
        try:
            xgb_model = xgb.XGBClassifier(n_estimators=10, random_state=42)
            xgb_model.fit(self.X, self.y)
            explainer = FraudModelExplainer(xgb_model)
            self.assertEqual(explainer.model_type, 'xgboost')
        except ImportError:
            self.skipTest("XGBoost not available")
        
        # Test LightGBM model
        try:
            lgb_model = lgb.LGBMClassifier(n_estimators=10, random_state=42)
            lgb_model.fit(self.X, self.y)
            explainer = FraudModelExplainer(lgb_model)
            self.assertEqual(explainer.model_type, 'lightgbm')
        except ImportError:
            self.skipTest("LightGBM not available")

class TestExplainabilityIntegration(unittest.TestCase):
    """Integration tests for explainability functionality"""
    
    def setUp(self):
        """Set up integration test data"""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 15
        
        # Create more realistic fraud detection data
        self.X = pd.DataFrame({
            'transaction_amount': np.random.exponential(100, self.n_samples),
            'time_since_signup': np.random.exponential(30, self.n_samples),
            'hour_of_day': np.random.randint(0, 24, self.n_samples),
            'day_of_week': np.random.randint(0, 7, self.n_samples),
            'user_age': np.random.normal(35, 10, self.n_samples),
            'location_risk': np.random.beta(2, 5, self.n_samples),
            'transaction_frequency': np.random.poisson(5, self.n_samples),
            'device_type_encoded': np.random.choice([0, 1, 2], self.n_samples),
            'ip_risk_score': np.random.uniform(0, 1, self.n_samples),
            'velocity_24h': np.random.exponential(10, self.n_samples),
            'velocity_7d': np.random.exponential(50, self.n_samples),
            'purchase_category_risk': np.random.beta(1, 3, self.n_samples),
            'user_trust_score': np.random.normal(0.7, 0.2, self.n_samples),
            'session_duration': np.random.exponential(300, self.n_samples),
            'browser_fingerprint_risk': np.random.uniform(0, 1, self.n_samples)
        })
        
        # Create realistic fraud target (imbalanced)
        fraud_prob = 1 / (1 + np.exp(-(
            -2 + 0.01 * self.X['transaction_amount'] + 
            0.1 * self.X['location_risk'] + 
            0.5 * self.X['ip_risk_score'] +
            -0.3 * self.X['user_trust_score']
        )))
        self.y = np.random.binomial(1, fraud_prob)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(self.X, self.y)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_explainability_workflow(self):
        """Test complete explainability workflow"""
        # Initialize explainer
        explainer = FraudModelExplainer(self.model)
        
        # Generate explanations
        results = explainer.explain_model(self.X, sample_size=100)
        
        # Get feature importance
        importance_df = explainer.get_feature_importance_ranking(10)
        
        # Analyze fraud drivers
        fraud_drivers = explainer.analyze_fraud_drivers(10)
        
        # Generate insights
        insights = explainer.interpret_fraud_patterns()
        
        # Create comprehensive report
        output_dir = os.path.join(self.temp_dir, 'integration_report')
        report = explainer.create_comprehensive_report(output_dir)
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertIsInstance(fraud_drivers, dict)
        self.assertIsInstance(insights, dict)
        self.assertIsInstance(report, dict)
        
        # Check that high-value transactions are identified as risk factors
        if 'transaction_amount' in fraud_drivers:
            self.assertIn('transaction_amount', [f['feature'] for f in insights['high_risk_features']])
    
    def test_business_insights_quality(self):
        """Test quality of business insights"""
        explainer = FraudModelExplainer(self.model)
        explainer.explain_model(self.X, sample_size=100)
        
        insights = explainer.interpret_fraud_patterns()
        
        # Check that insights are meaningful
        self.assertGreater(len(insights['high_risk_features']), 0)
        self.assertGreater(len(insights['recommendations']), 0)
        
        # Check that recommendations are actionable
        for recommendation in insights['recommendations']:
            self.assertIsInstance(recommendation, str)
            self.assertGreater(len(recommendation), 10)
    
    def test_plot_generation_quality(self):
        """Test quality of generated plots"""
        explainer = FraudModelExplainer(self.model)
        explainer.explain_model(self.X, sample_size=100)
        
        # Generate all types of plots
        plots = {}
        
        # Summary plot
        plots['summary'] = explainer.create_summary_plot(show=False)
        
        # Force plot
        plots['force'] = explainer.create_force_plot(instance_idx=0, show=False)
        
        # Waterfall plot
        plots['waterfall'] = explainer.create_waterfall_plot(instance_idx=0, show=False)
        
        # Dependence plot
        plots['dependence'] = explainer.create_dependence_plot('transaction_amount', show=False)
        
        # Verify all plots were created
        for plot_name, plot in plots.items():
            self.assertIsNotNone(plot, f"Plot {plot_name} was not created")

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 