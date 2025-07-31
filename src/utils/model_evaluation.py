"""
Model Evaluation Utilities for Fraud Detection

This module provides comprehensive evaluation functions for fraud detection models,
including fraud-specific metrics, visualizations, and business impact analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FraudModelEvaluator:
    """
    Comprehensive model evaluator for fraud detection.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
        self.business_metrics = {}
        
    def calculate_fraud_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                              y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive fraud detection metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of fraud-specific metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC metrics
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional fraud-specific metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # False positive and negative rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Positive and negative predictive values
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'matthews_correlation': mcc,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        return metrics
    
    def evaluate_model_performance(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                 model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate model performance comprehensively.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_fraud_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test
        }
        
        self.evaluation_results[model_name] = results
        
        # Log key metrics
        logger.info(f"{model_name} - F1: {metrics['f1_score']:.4f}, "
                   f"PR-AUC: {metrics['pr_auc']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return results
    
    def compare_models(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                      y_test: pd.Series) -> pd.DataFrame:
        """
        Compare multiple models and return comparison table.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame with model comparison
        """
        logger.info("Comparing multiple models...")
        
        comparison_data = []
        
        for model_name, model in models.items():
            results = self.evaluate_model_performance(model, X_test, y_test, model_name)
            metrics = results['metrics']
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc'],
                'PR-AUC': metrics['pr_auc'],
                'Specificity': metrics['specificity'],
                'Sensitivity': metrics['sensitivity']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def create_comprehensive_plots(self, save_path: str = None) -> None:
        """
        Create comprehensive evaluation plots.
        
        Args:
            save_path: Path to save plots (optional)
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available. Run evaluate_model_performance() first.")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Model Evaluation', fontsize=16, fontweight='bold')
        
        # Colors for different models
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # 1. ROC Curves
        ax = axes[0, 0]
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            fpr, tpr, _ = roc_curve(results['true_labels'], results['probabilities'])
            auc = results['metrics']['roc_auc']
            ax.plot(fpr, tpr, color=colors[i % len(colors)], 
                   label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curves
        ax = axes[0, 1]
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            precision, recall, _ = precision_recall_curve(results['true_labels'], results['probabilities'])
            pr_auc = results['metrics']['pr_auc']
            ax.plot(recall, precision, color=colors[i % len(colors)], 
                   label=f'{model_name} (PR-AUC = {pr_auc:.3f})', linewidth=2)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Metrics Comparison
        ax = axes[0, 2]
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(self.evaluation_results)
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            values = [results['metrics'][metric] for metric in metrics_to_plot]
            ax.bar(x + i * width, values, width, label=model_name, 
                  color=colors[i % len(colors)], alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Metrics Comparison')
        ax.set_xticks(x + width * (len(self.evaluation_results) - 1) / 2)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Confusion Matrices
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            if i >= 3:  # Limit to 3 models for confusion matrices
                break
                
            row = 1
            col = i
            ax = axes[row, col]
            cm = results['confusion_matrix']
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # 5. Business Impact Metrics
        ax = axes[1, 2]
        business_metrics = ['false_positive_rate', 'false_negative_rate', 'specificity', 'sensitivity']
        x = np.arange(len(business_metrics))
        width = 0.8 / len(self.evaluation_results)
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            values = [results['metrics'][metric] for metric in business_metrics]
            ax.bar(x + i * width, values, width, label=model_name, 
                  color=colors[i % len(colors)], alpha=0.8)
        
        ax.set_xlabel('Business Metrics')
        ax.set_ylabel('Rate')
        ax.set_title('Business Impact Metrics')
        ax.set_xticks(x + width * (len(self.evaluation_results) - 1) / 2)
        ax.set_xticklabels(['FPR', 'FNR', 'Specificity', 'Sensitivity'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {save_path}")
        
        plt.show()
    
    def analyze_business_impact(self, model_name: str, 
                              fraud_cost: float = 100.0, 
                              false_positive_cost: float = 10.0) -> Dict[str, float]:
        """
        Analyze business impact of model performance.
        
        Args:
            model_name: Name of the model to analyze
            fraud_cost: Cost of a missed fraud (false negative)
            false_positive_cost: Cost of a false positive
            
        Returns:
            Dictionary with business impact metrics
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")
        
        results = self.evaluation_results[model_name]
        metrics = results['metrics']
        
        # Extract confusion matrix values
        tp = metrics['true_positives']
        tn = metrics['true_negatives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']
        
        # Calculate business metrics
        total_transactions = tp + tn + fp + fn
        fraud_rate = (tp + fn) / total_transactions if total_transactions > 0 else 0
        
        # Cost analysis
        fraud_prevention_savings = tp * fraud_cost  # Money saved by catching fraud
        false_positive_costs = fp * false_positive_cost  # Cost of false positives
        missed_fraud_costs = fn * fraud_cost  # Cost of missed fraud
        
        net_savings = fraud_prevention_savings - false_positive_costs - missed_fraud_costs
        roi = (net_savings / (false_positive_costs + missed_fraud_costs)) * 100 if (false_positive_costs + missed_fraud_costs) > 0 else float('inf')
        
        # Risk metrics
        fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        business_impact = {
            'total_transactions': total_transactions,
            'fraud_rate': fraud_rate,
            'fraud_detection_rate': fraud_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'fraud_prevention_savings': fraud_prevention_savings,
            'false_positive_costs': false_positive_costs,
            'missed_fraud_costs': missed_fraud_costs,
            'net_savings': net_savings,
            'roi_percentage': roi,
            'cost_per_transaction': (false_positive_costs + missed_fraud_costs) / total_transactions if total_transactions > 0 else 0
        }
        
        self.business_metrics[model_name] = business_impact
        return business_impact
    
    def generate_evaluation_report(self, model_name: str = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            model_name: Specific model to report on (if None, report on all)
            
        Returns:
            Formatted evaluation report
        """
        if not self.evaluation_results:
            return "No evaluation results available."
        
        report = []
        report.append("=" * 80)
        report.append("FRAUD DETECTION MODEL EVALUATION REPORT")
        report.append("=" * 80)
        
        if model_name:
            models_to_report = [model_name] if model_name in self.evaluation_results else []
        else:
            models_to_report = list(self.evaluation_results.keys())
        
        for model_name in models_to_report:
            results = self.evaluation_results[model_name]
            metrics = results['metrics']
            
            report.append(f"\n🎯 MODEL: {model_name.upper()}")
            report.append("-" * 40)
            
            # Performance metrics
            report.append("\n📊 PERFORMANCE METRICS:")
            report.append(f"  Accuracy: {metrics['accuracy']:.4f}")
            report.append(f"  Precision: {metrics['precision']:.4f}")
            report.append(f"  Recall: {metrics['recall']:.4f}")
            report.append(f"  F1-Score: {metrics['f1_score']:.4f}")
            report.append(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            report.append(f"  PR-AUC: {metrics['pr_auc']:.4f}")
            
            # Fraud-specific metrics
            report.append("\n🛡️ FRAUD-SPECIFIC METRICS:")
            report.append(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            report.append(f"  Specificity: {metrics['specificity']:.4f}")
            report.append(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
            report.append(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
            
            # Confusion matrix
            cm = results['confusion_matrix']
            report.append("\n📋 CONFUSION MATRIX:")
            report.append(f"  True Negatives: {cm[0,0]}")
            report.append(f"  False Positives: {cm[0,1]}")
            report.append(f"  False Negatives: {cm[1,0]}")
            report.append(f"  True Positives: {cm[1,1]}")
            
            # Business impact
            if model_name in self.business_metrics:
                business = self.business_metrics[model_name]
                report.append("\n💼 BUSINESS IMPACT:")
                report.append(f"  Fraud Detection Rate: {business['fraud_detection_rate']:.2%}")
                report.append(f"  False Alarm Rate: {business['false_alarm_rate']:.2%}")
                report.append(f"  Net Savings: ${business['net_savings']:,.2f}")
                report.append(f"  ROI: {business['roi_percentage']:.2f}%")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def get_best_model(self, metric: str = 'f1_score') -> str:
        """
        Get the best model based on specified metric.
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            Name of the best model
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        best_score = -1
        best_model = None
        
        for model_name, results in self.evaluation_results.items():
            score = results['metrics'][metric]
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model


def main():
    """Example usage of the FraudModelEvaluator class."""
    
    # Initialize evaluator
    evaluator = FraudModelEvaluator()
    
    # Example workflow
    print("Fraud Model Evaluator initialized successfully!")
    print("Use this class to:")
    print("1. Calculate comprehensive fraud detection metrics")
    print("2. Compare multiple models")
    print("3. Analyze business impact")
    print("4. Generate evaluation reports")
    print("5. Create visualization plots")


if __name__ == "__main__":
    main() 