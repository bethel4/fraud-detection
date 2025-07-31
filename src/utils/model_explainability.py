"""
Model Explainability Module for Fraud Detection
Uses SHAP (Shapley Additive exPlanations) to interpret fraud detection models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from pathlib import Path

class FraudModelExplainer:
    """
    SHAP-based explainability for fraud detection models
    """
    
    def __init__(self, model, model_type: str = 'auto', random_state: int = 42):
        """
        Initialize the explainer
        
        Args:
            model: Trained model (RandomForest, XGBoost, LightGBM, LogisticRegression)
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm', 'logistic', 'auto')
            random_state: Random state for reproducibility
        """
        self.model = model
        self.model_type = self._detect_model_type() if model_type == 'auto' else model_type
        self.random_state = random_state
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
        # Initialize SHAP explainer
        self._initialize_explainer()
    
    def _detect_model_type(self) -> str:
        """Automatically detect model type"""
        model_class = type(self.model).__name__
        
        if 'RandomForest' in model_class:
            return 'random_forest'
        elif 'XGB' in model_class or 'XGBoost' in model_class:
            return 'xgboost'
        elif 'LGBM' in model_class or 'LightGBM' in model_class:
            return 'lightgbm'
        elif 'Logistic' in model_class:
            return 'logistic'
        else:
            return 'tree'  # Default for tree-based models
    
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer"""
        try:
            if self.model_type == 'random_forest':
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == 'xgboost':
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == 'lightgbm':
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == 'logistic':
                # For linear models, we'll use KernelExplainer
                self.explainer = None  # Will be initialized when needed
            else:
                self.explainer = shap.TreeExplainer(self.model)
                
            print(f"✅ SHAP explainer initialized for {self.model_type} model")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize explainer: {e}")
            self.explainer = None
    
    def explain_model(self, X: pd.DataFrame, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate comprehensive SHAP explanations
        
        Args:
            X: Feature matrix
            sample_size: Number of samples to use (None for all)
            
        Returns:
            Dictionary containing SHAP values and explanations
        """
        print("🔍 Generating SHAP explanations...")
        
        # Sample data if specified
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=self.random_state)
            print(f"📊 Using {sample_size} samples for explanation")
        else:
            X_sample = X
            print(f"📊 Using all {len(X)} samples for explanation")
        
        # Generate SHAP values
        if self.model_type == 'logistic':
            # Use KernelExplainer for linear models
            background = X_sample.sample(min(100, len(X_sample)), random_state=self.random_state)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            self.shap_values = self.explainer.shap_values(X_sample)
            self.expected_value = self.explainer.expected_value
        else:
            # Use TreeExplainer for tree-based models
            self.shap_values = self.explainer.shap_values(X_sample)
            self.expected_value = self.explainer.expected_value
        
        # Handle different SHAP value formats
        if isinstance(self.shap_values, list):
            # For classification, use the positive class (fraud)
            self.shap_values = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        
        print("✅ SHAP values generated successfully")
        
        return {
            'shap_values': self.shap_values,
            'expected_value': self.expected_value,
            'feature_names': X_sample.columns.tolist(),
            'sample_data': X_sample
        }
    
    def create_summary_plot(self, save_path: Optional[str] = None, 
                           max_display: int = 20, show: bool = True) -> plt.Figure:
        """
        Create SHAP summary plot showing global feature importance
        
        Args:
            save_path: Path to save the plot
            max_display: Maximum number of features to display
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not generated. Run explain_model() first.")
        
        print("📊 Creating SHAP summary plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        shap.summary_plot(
            self.shap_values, 
            self.sample_data,
            max_display=max_display,
            show=False,
            plot_type="bar"
        )
        
        plt.title("SHAP Feature Importance - Fraud Detection Model", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("mean(|SHAP value|)", fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Summary plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def create_force_plot(self, instance_idx: int = 0, save_path: Optional[str] = None,
                         show: bool = True) -> plt.Figure:
        """
        Create SHAP force plot for a specific instance
        
        Args:
            instance_idx: Index of the instance to explain
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not generated. Run explain_model() first.")
        
        print(f"🔍 Creating force plot for instance {instance_idx}...")
        
        fig = plt.figure(figsize=(12, 6))
        
        shap.force_plot(
            self.expected_value,
            self.shap_values[instance_idx, :],
            self.sample_data.iloc[instance_idx, :],
            show=False,
            matplotlib=True
        )
        
        plt.title(f"SHAP Force Plot - Instance {instance_idx}", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Force plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def create_waterfall_plot(self, instance_idx: int = 0, save_path: Optional[str] = None,
                             show: bool = True) -> plt.Figure:
        """
        Create SHAP waterfall plot for a specific instance
        
        Args:
            instance_idx: Index of the instance to explain
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not generated. Run explain_model() first.")
        
        print(f"🌊 Creating waterfall plot for instance {instance_idx}...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_idx, :],
                base_values=self.expected_value,
                data=self.sample_data.iloc[instance_idx, :].values,
                feature_names=self.sample_data.columns.tolist()
            ),
            show=False
        )
        
        plt.title(f"SHAP Waterfall Plot - Instance {instance_idx}", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Waterfall plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def create_dependence_plot(self, feature_name: str, interaction_index: Optional[str] = None,
                              save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Create SHAP dependence plot for a specific feature
        
        Args:
            feature_name: Name of the feature to analyze
            interaction_index: Feature to use for interaction coloring
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not generated. Run explain_model() first.")
        
        print(f"📈 Creating dependence plot for feature: {feature_name}...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        shap.dependence_plot(
            feature_name,
            self.shap_values,
            self.sample_data,
            interaction_index=interaction_index,
            show=False
        )
        
        plt.title(f"SHAP Dependence Plot - {feature_name}", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Dependence plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def create_interaction_plot(self, feature1: str, feature2: str,
                               save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Create SHAP interaction plot between two features
        
        Args:
            feature1: First feature name
            feature2: Second feature name
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not generated. Run explain_model() first.")
        
        print(f"🔄 Creating interaction plot for {feature1} vs {feature2}...")
        
        # Calculate interaction values
        interaction_values = shap.TreeExplainer(self.model).shap_interaction_values(self.sample_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        shap.dependence_plot(
            (feature1, feature2),
            interaction_values,
            self.sample_data,
            show=False
        )
        
        plt.title(f"SHAP Interaction Plot - {feature1} vs {feature2}", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Interaction plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def get_feature_importance_ranking(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get ranked feature importance based on SHAP values
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance ranking
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not generated. Run explain_model() first.")
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create ranking DataFrame
        importance_df = pd.DataFrame({
            'feature': self.sample_data.columns,
            'mean_abs_shap': mean_shap,
            'std_shap': np.std(self.shap_values, axis=0)
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
        
        return importance_df.head(top_n)
    
    def analyze_fraud_drivers(self, top_features: int = 10) -> Dict[str, Any]:
        """
        Analyze key drivers of fraud based on SHAP values
        
        Args:
            top_features: Number of top features to analyze
            
        Returns:
            Dictionary with fraud driver analysis
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not generated. Run explain_model() first.")
        
        print("🔍 Analyzing fraud drivers...")
        
        # Get feature importance ranking
        importance_df = self.get_feature_importance_ranking(top_features)
        
        # Analyze feature effects
        fraud_drivers = {}
        
        for _, row in importance_df.iterrows():
            feature = row['feature']
            feature_idx = self.sample_data.columns.get_loc(feature)
            
            # Get SHAP values for this feature
            feature_shap = self.shap_values[:, feature_idx]
            feature_values = self.sample_data[feature]
            
            # Calculate statistics
            positive_effect = np.mean(feature_shap[feature_shap > 0])
            negative_effect = np.mean(feature_shap[feature_shap < 0])
            
            fraud_drivers[feature] = {
                'importance': row['mean_abs_shap'],
                'std_importance': row['std_shap'],
                'positive_effect': positive_effect,
                'negative_effect': negative_effect,
                'mean_value': feature_values.mean(),
                'std_value': feature_values.std(),
                'effect_direction': 'positive' if positive_effect > abs(negative_effect) else 'negative'
            }
        
        return fraud_drivers
    
    def create_comprehensive_report(self, output_dir: str = "results/explainability") -> Dict[str, Any]:
        """
        Create comprehensive explainability report with all plots and analysis
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Dictionary with report summary
        """
        print("📊 Creating comprehensive explainability report...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all plots
        plots = {}
        
        # Summary plot
        summary_plot_path = os.path.join(output_dir, "shap_summary_plot.png")
        plots['summary'] = self.create_summary_plot(save_path=summary_plot_path, show=False)
        
        # Force plots for top instances
        for i in range(min(5, len(self.sample_data))):
            force_plot_path = os.path.join(output_dir, f"shap_force_plot_instance_{i}.png")
            plots[f'force_{i}'] = self.create_force_plot(i, save_path=force_plot_path, show=False)
        
        # Waterfall plots for top instances
        for i in range(min(3, len(self.sample_data))):
            waterfall_plot_path = os.path.join(output_dir, f"shap_waterfall_plot_instance_{i}.png")
            plots[f'waterfall_{i}'] = self.create_waterfall_plot(i, save_path=waterfall_plot_path, show=False)
        
        # Dependence plots for top features
        importance_df = self.get_feature_importance_ranking(5)
        for feature in importance_df['feature']:
            dep_plot_path = os.path.join(output_dir, f"shap_dependence_plot_{feature}.png")
            plots[f'dependence_{feature}'] = self.create_dependence_plot(
                feature, save_path=dep_plot_path, show=False
            )
        
        # Analyze fraud drivers
        fraud_drivers = self.analyze_fraud_drivers()
        
        # Create report summary
        report = {
            'plots_generated': len(plots),
            'output_directory': output_dir,
            'fraud_drivers': fraud_drivers,
            'feature_importance': self.get_feature_importance_ranking().to_dict('records'),
            'model_type': self.model_type,
            'samples_analyzed': len(self.sample_data)
        }
        
        # Save report to file
        report_path = os.path.join(output_dir, "explainability_report.txt")
        with open(report_path, 'w') as f:
            f.write("FRAUD DETECTION MODEL EXPLAINABILITY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Samples Analyzed: {len(self.sample_data)}\n")
            f.write(f"Plots Generated: {len(plots)}\n\n")
            
            f.write("TOP FRAUD DRIVERS:\n")
            f.write("-" * 20 + "\n")
            for feature, analysis in fraud_drivers.items():
                f.write(f"\n{feature}:\n")
                f.write(f"  Importance: {analysis['importance']:.4f}\n")
                f.write(f"  Effect Direction: {analysis['effect_direction']}\n")
                f.write(f"  Positive Effect: {analysis['positive_effect']:.4f}\n")
                f.write(f"  Negative Effect: {analysis['negative_effect']:.4f}\n")
        
        print(f"✅ Comprehensive report saved to {output_dir}")
        
        return report
    
    def interpret_fraud_patterns(self) -> Dict[str, str]:
        """
        Provide business interpretation of fraud patterns
        
        Returns:
            Dictionary with business insights
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not generated. Run explain_model() first.")
        
        fraud_drivers = self.analyze_fraud_drivers()
        
        insights = {
            'high_risk_features': [],
            'protective_features': [],
            'behavioral_patterns': [],
            'recommendations': []
        }
        
        for feature, analysis in fraud_drivers.items():
            if analysis['effect_direction'] == 'positive':
                insights['high_risk_features'].append({
                    'feature': feature,
                    'importance': analysis['importance'],
                    'interpretation': f"Higher values of {feature} increase fraud risk"
                })
            else:
                insights['protective_features'].append({
                    'feature': feature,
                    'importance': analysis['importance'],
                    'interpretation': f"Higher values of {feature} decrease fraud risk"
                })
        
        # Generate recommendations
        if insights['high_risk_features']:
            insights['recommendations'].append(
                "Monitor transactions with high values in risk features"
            )
        
        if insights['protective_features']:
            insights['recommendations'].append(
                "Encourage behaviors associated with protective features"
            )
        
        return insights 