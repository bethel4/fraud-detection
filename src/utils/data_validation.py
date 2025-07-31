"""
Data Validation Utilities for Fraud Detection

This module provides validation functions to ensure data quality and consistency
throughout the preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Data validation class for fraud detection datasets.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.validation_results = {}
    
    def validate_fraud_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate fraud detection dataset structure and content.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating fraud detection dataset...")
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality_score': 0.0
        }
        
        # Check required columns
        required_columns = ['user_id', 'purchase_time', 'purchase_value', 'class']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
            validation_results['is_valid'] = False
        
        # Check data types
        expected_dtypes = {
            'user_id': ['int64', 'object'],
            'purchase_value': ['float64', 'int64'],
            'class': ['int64', 'object']
        }
        
        for col, expected_types in expected_dtypes.items():
            if col in df.columns:
                if str(df[col].dtype) not in expected_types:
                    validation_results['warnings'].append(
                        f"Column {col} has unexpected dtype: {df[col].dtype}"
                    )
        
        # Check for negative purchase values
        if 'purchase_value' in df.columns:
            negative_values = (df['purchase_value'] < 0).sum()
            if negative_values > 0:
                validation_results['warnings'].append(
                    f"Found {negative_values} negative purchase values"
                )
        
        # Check target distribution
        if 'class' in df.columns:
            class_counts = df['class'].value_counts()
            if len(class_counts) != 2:
                validation_results['errors'].append(
                    f"Target column should have exactly 2 classes, found {len(class_counts)}"
                )
                validation_results['is_valid'] = False
            
            # Check for extreme class imbalance
            min_class_ratio = min(class_counts) / max(class_counts)
            if min_class_ratio < 0.01:
                validation_results['warnings'].append(
                    f"Extreme class imbalance detected: {min_class_ratio:.4f}"
                )
        
        # Calculate data quality score
        validation_results['data_quality_score'] = self._calculate_quality_score(df)
        
        self.validation_results = validation_results
        return validation_results
    
    def validate_ip_country_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate IP country mapping dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating IP country mapping dataset...")
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality_score': 0.0
        }
        
        # Check required columns
        required_columns = ['ip_address', 'country']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
            validation_results['is_valid'] = False
        
        # Check IP address format
        if 'ip_address' in df.columns:
            invalid_ips = 0
            for ip in df['ip_address'].dropna():
                if not self._is_valid_ip(ip):
                    invalid_ips += 1
            
            if invalid_ips > 0:
                validation_results['warnings'].append(
                    f"Found {invalid_ips} invalid IP addresses"
                )
        
        # Check for duplicate IP addresses
        if 'ip_address' in df.columns:
            duplicates = df['ip_address'].duplicated().sum()
            if duplicates > 0:
                validation_results['warnings'].append(
                    f"Found {duplicates} duplicate IP addresses"
                )
        
        # Calculate data quality score
        validation_results['data_quality_score'] = self._calculate_quality_score(df)
        
        return validation_results
    
    def _is_valid_ip(self, ip_address: str) -> bool:
        """
        Check if IP address is valid.
        
        Args:
            ip_address: IP address string
            
        Returns:
            True if valid, False otherwise
        """
        try:
            parts = ip_address.split('.')
            if len(parts) != 4:
                return False
            
            for part in parts:
                if not part.isdigit():
                    return False
                num = int(part)
                if num < 0 or num > 255:
                    return False
            
            return True
        except:
            return False
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate data quality score based on various metrics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Penalize for missing values
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        score -= missing_ratio * 0.3
        
        # Penalize for duplicates
        duplicate_ratio = df.duplicated().sum() / df.shape[0]
        score -= duplicate_ratio * 0.2
        
        # Penalize for extreme outliers in numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_ratio = outliers / df.shape[0]
            score -= outlier_ratio * 0.1
        
        return max(0.0, min(1.0, score))
    
    def generate_validation_report(self) -> str:
        """
        Generate a comprehensive validation report.
        
        Returns:
            Formatted validation report string
        """
        if not self.validation_results:
            return "No validation results available."
        
        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        
        # Overall status
        status = "✅ VALID" if self.validation_results['is_valid'] else "❌ INVALID"
        report.append(f"Overall Status: {status}")
        report.append(f"Data Quality Score: {self.validation_results['data_quality_score']:.2%}")
        report.append("")
        
        # Errors
        if self.validation_results['errors']:
            report.append("❌ ERRORS:")
            for error in self.validation_results['errors']:
                report.append(f"  • {error}")
            report.append("")
        
        # Warnings
        if self.validation_results['warnings']:
            report.append("⚠️  WARNINGS:")
            for warning in self.validation_results['warnings']:
                report.append(f"  • {warning}")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        if not self.validation_results['is_valid']:
            report.append("  • Fix all errors before proceeding")
        if self.validation_results['warnings']:
            report.append("  • Review warnings and address critical issues")
        if self.validation_results['data_quality_score'] < 0.8:
            report.append("  • Consider additional data cleaning steps")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def validate_data_consistency(df1: pd.DataFrame, df2: pd.DataFrame, 
                            key_column: str) -> Dict:
    """
    Validate consistency between two datasets.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        key_column: Column to use for comparison
        
    Returns:
        Dictionary with consistency validation results
    """
    logger.info(f"Validating data consistency using key column: {key_column}")
    
    results = {
        'is_consistent': True,
        'errors': [],
        'warnings': [],
        'overlap_ratio': 0.0
    }
    
    # Check if key column exists in both datasets
    if key_column not in df1.columns:
        results['errors'].append(f"Key column '{key_column}' not found in first dataset")
        results['is_consistent'] = False
    
    if key_column not in df2.columns:
        results['errors'].append(f"Key column '{key_column}' not found in second dataset")
        results['is_consistent'] = False
    
    if not results['is_consistent']:
        return results
    
    # Calculate overlap
    keys1 = set(df1[key_column].dropna())
    keys2 = set(df2[key_column].dropna())
    
    overlap = keys1.intersection(keys2)
    total_unique = keys1.union(keys2)
    
    if total_unique:
        results['overlap_ratio'] = len(overlap) / len(total_unique)
    
    # Check for consistency issues
    if results['overlap_ratio'] < 0.5:
        results['warnings'].append(
            f"Low overlap ratio: {results['overlap_ratio']:.2%}"
        )
    
    if len(keys1 - keys2) > 0:
        results['warnings'].append(
            f"Keys in first dataset but not in second: {len(keys1 - keys2)}"
        )
    
    if len(keys2 - keys1) > 0:
        results['warnings'].append(
            f"Keys in second dataset but not in first: {len(keys2 - keys1)}"
        )
    
    return results


def check_feature_quality(features: pd.DataFrame, target: pd.Series) -> Dict:
    """
    Check quality of engineered features.
    
    Args:
        features: Feature DataFrame
        target: Target Series
        
    Returns:
        Dictionary with feature quality metrics
    """
    logger.info("Checking feature quality...")
    
    quality_metrics = {
        'total_features': len(features.columns),
        'numerical_features': 0,
        'categorical_features': 0,
        'high_correlation_features': 0,
        'low_variance_features': 0,
        'missing_value_features': 0,
        'recommendations': []
    }
    
    # Count feature types
    numerical_features = features.select_dtypes(include=[np.number]).columns
    categorical_features = features.select_dtypes(include=['object', 'category']).columns
    
    quality_metrics['numerical_features'] = len(numerical_features)
    quality_metrics['categorical_features'] = len(categorical_features)
    
    # Check for high correlation features
    if len(numerical_features) > 1:
        correlation_matrix = features[numerical_features].corr()
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.95:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        quality_metrics['high_correlation_features'] = len(high_corr_pairs)
        
        if high_corr_pairs:
            quality_metrics['recommendations'].append(
                f"Consider removing one of {len(high_corr_pairs)} highly correlated feature pairs"
            )
    
    # Check for low variance features
    for col in numerical_features:
        if features[col].std() < 0.01:
            quality_metrics['low_variance_features'] += 1
    
    if quality_metrics['low_variance_features'] > 0:
        quality_metrics['recommendations'].append(
            f"Consider removing {quality_metrics['low_variance_features']} low variance features"
        )
    
    # Check for missing values
    for col in features.columns:
        if features[col].isnull().sum() > 0:
            quality_metrics['missing_value_features'] += 1
    
    if quality_metrics['missing_value_features'] > 0:
        quality_metrics['recommendations'].append(
            f"Handle missing values in {quality_metrics['missing_value_features']} features"
        )
    
    # Calculate feature importance correlation with target
    if len(numerical_features) > 0:
        correlations = features[numerical_features].corrwith(target).abs()
        strong_correlations = (correlations > 0.1).sum()
        quality_metrics['strong_target_correlations'] = strong_correlations
        
        if strong_correlations < len(numerical_features) * 0.1:
            quality_metrics['recommendations'].append(
                "Consider feature selection as few features correlate strongly with target"
            )
    
    return quality_metrics


def main():
    """Example usage of data validation functions."""
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'user_id': range(1000),
        'purchase_value': np.random.exponential(100, 1000),
        'class': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    })
    
    # Initialize validator
    validator = DataValidator()
    
    # Validate data
    results = validator.validate_fraud_data(sample_data)
    
    # Generate report
    report = validator.generate_validation_report()
    print(report)


if __name__ == "__main__":
    main() 