"""
Feature Engineering Module for Fraud Detection

This module contains functions to create features for both e-commerce and credit card fraud detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudFeatureEngineer:
    """
    Feature engineering class for fraud detection datasets.
    
    Handles both e-commerce fraud (Fraud_Data.csv) and credit card fraud (creditcard.csv).
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.ip_country_mapping = None
        self.user_transaction_history = {}
        
    def load_ip_country_mapping(self, ip_country_file: str) -> None:
        """
        Load IP address to country mapping.
        
        Args:
            ip_country_file: Path to IpAddress_to_Country.csv
        """
        try:
            self.ip_country_mapping = pd.read_csv(ip_country_file)
            logger.info(f"Loaded IP country mapping with {len(self.ip_country_mapping)} records")
        except Exception as e:
            logger.error(f"Error loading IP country mapping: {e}")
            self.ip_country_mapping = None
    
    def create_ecommerce_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for e-commerce fraud detection.
        
        Args:
            df: DataFrame with e-commerce transaction data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating e-commerce fraud features...")
        
        # Create a copy to avoid modifying original data
        df_features = df.copy()
        
        # Convert timestamps to datetime
        df_features['signup_time'] = pd.to_datetime(df_features['signup_time'])
        df_features['purchase_time'] = pd.to_datetime(df_features['purchase_time'])
        
        # Time-based features
        df_features = self._create_time_features(df_features)
        
        # User behavior features
        df_features = self._create_user_behavior_features(df_features)
        
        # Device and browser features
        df_features = self._create_device_features(df_features)
        
        # Location-based features
        if self.ip_country_mapping is not None:
            df_features = self._create_location_features(df_features)
        
        # Transaction value features
        df_features = self._create_transaction_features(df_features)
        
        logger.info(f"Created {len(df_features.columns)} features for e-commerce data")
        return df_features
    
    def create_creditcard_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for credit card fraud detection.
        
        Args:
            df: DataFrame with credit card transaction data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating credit card fraud features...")
        
        # Create a copy to avoid modifying original data
        df_features = df.copy()
        
        # Amount-based features
        df_features = self._create_amount_features(df_features)
        
        # Time-based features
        df_features = self._create_creditcard_time_features(df_features)
        
        # Statistical features from PCA components
        df_features = self._create_pca_statistical_features(df_features)
        
        # Interaction features between amount and PCA components
        df_features = self._create_interaction_features(df_features)
        
        logger.info(f"Created {len(df_features.columns)} features for credit card data")
        return df_features
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features for e-commerce data."""
        
        # Time difference between signup and purchase
        df['signup_to_purchase_hours'] = (
            df['purchase_time'] - df['signup_time']
        ).dt.total_seconds() / 3600
        
        # Purchase time features
        df['purchase_hour'] = df['purchase_time'].dt.hour
        df['purchase_day_of_week'] = df['purchase_time'].dt.dayofweek
        df['purchase_month'] = df['purchase_time'].dt.month
        df['purchase_day'] = df['purchase_time'].dt.day
        
        # Signup time features
        df['signup_hour'] = df['signup_time'].dt.hour
        df['signup_day_of_week'] = df['signup_time'].dt.dayofweek
        
        # Time-based flags
        df['is_night_purchase'] = (df['purchase_hour'] >= 22) | (df['purchase_hour'] <= 6)
        df['is_weekend_purchase'] = df['purchase_day_of_week'].isin([5, 6])
        
        return df
    
    def _create_user_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user behavior features."""
        
        # User age features
        df['user_age'] = df['age']
        df['is_young_user'] = df['age'] < 25
        df['is_senior_user'] = df['age'] > 65
        
        # Gender encoding
        df['gender_encoded'] = df['gender'].map({'M': 1, 'F': 0})
        
        # Quick purchase after signup (potential fraud indicator)
        df['quick_purchase_flag'] = df['signup_to_purchase_hours'] < 1
        
        return df
    
    def _create_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create device and browser features."""
        
        # Device type encoding
        device_encoding = {
            'mobile': 1,
            'desktop': 2,
            'tablet': 3
        }
        df['device_encoded'] = df['device_id'].map(device_encoding)
        
        # Browser encoding
        browser_encoding = {
            'chrome': 1,
            'firefox': 2,
            'safari': 3,
            'edge': 4,
            'opera': 5
        }
        df['browser_encoded'] = df['browser'].map(browser_encoding)
        
        # Source encoding
        source_encoding = {
            'direct': 1,
            'organic': 2,
            'paid': 3,
            'referral': 4
        }
        df['source_encoded'] = df['source'].map(source_encoding)
        
        return df
    
    def _create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based features using IP country mapping."""
        
        # Merge with IP country mapping
        df = df.merge(
            self.ip_country_mapping[['ip_address', 'country']], 
            on='ip_address', 
            how='left'
        )
        
        # Country encoding (simple hash-based encoding)
        df['country_encoded'] = df['country'].astype('category').cat.codes
        
        # Missing country flag
        df['missing_country'] = df['country'].isna().astype(int)
        
        return df
    
    def _create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transaction value features."""
        
        # Transaction amount features
        df['purchase_value'] = df['purchase_value']
        df['log_purchase_value'] = np.log1p(df['purchase_value'])
        
        # High value transaction flag
        df['is_high_value'] = df['purchase_value'] > df['purchase_value'].quantile(0.95)
        
        # Low value transaction flag
        df['is_low_value'] = df['purchase_value'] < df['purchase_value'].quantile(0.05)
        
        return df
    
    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features for credit card data."""
        
        # Log transformation of amount
        df['log_amount'] = np.log1p(df['Amount'])
        
        # Amount statistics
        df['amount_mean'] = df['Amount'].mean()
        df['amount_std'] = df['Amount'].std()
        df['amount_median'] = df['Amount'].median()
        
        # Amount percentiles
        df['amount_percentile'] = df['Amount'].rank(pct=True)
        
        # High and low amount flags
        df['is_high_amount'] = df['Amount'] > df['Amount'].quantile(0.95)
        df['is_low_amount'] = df['Amount'] < df['Amount'].quantile(0.05)
        
        return df
    
    def _create_creditcard_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features for credit card data."""
        
        # Time features (assuming Time column is in seconds)
        df['time_hour'] = (df['Time'] // 3600) % 24
        df['time_day'] = (df['Time'] // 86400) % 7
        
        # Time-based flags
        df['is_night_time'] = (df['time_hour'] >= 22) | (df['time_hour'] <= 6)
        df['is_weekend'] = df['time_day'].isin([5, 6])
        
        return df
    
    def _create_pca_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from PCA components."""
        
        # Get PCA columns (V1 to V28)
        pca_cols = [col for col in df.columns if col.startswith('V')]
        
        if pca_cols:
            # Statistical features across PCA components
            df['pca_mean'] = df[pca_cols].mean(axis=1)
            df['pca_std'] = df[pca_cols].std(axis=1)
            df['pca_max'] = df[pca_cols].max(axis=1)
            df['pca_min'] = df[pca_cols].min(axis=1)
            df['pca_range'] = df['pca_max'] - df['pca_min']
            
            # Count of extreme values
            df['pca_extreme_count'] = (df[pca_cols] > 3).sum(axis=1)
            
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between amount and PCA components."""
        
        # Get PCA columns
        pca_cols = [col for col in df.columns if col.startswith('V')]
        
        if pca_cols:
            # Interaction between amount and PCA components
            for i, col in enumerate(pca_cols[:5]):  # Use first 5 PCA components
                df[f'amount_pca_{i+1}_interaction'] = df['Amount'] * df[col]
        
        return df
    
    def get_feature_importance_ranking(self, df: pd.DataFrame, target_col: str = 'Class') -> pd.DataFrame:
        """
        Get feature importance ranking using correlation with target.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            DataFrame with feature importance ranking
        """
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in DataFrame")
            return pd.DataFrame()
        
        # Calculate correlation with target
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        
        # Remove target column from ranking
        correlations = correlations[correlations.index != target_col]
        
        # Create ranking DataFrame
        ranking = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values,
            'rank': range(1, len(correlations) + 1)
        })
        
        return ranking
    
    def select_top_features(self, df: pd.DataFrame, target_col: str = 'Class', 
                          top_n: int = 20) -> List[str]:
        """
        Select top N most important features.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            top_n: Number of top features to select
            
        Returns:
            List of top feature names
        """
        ranking = self.get_feature_importance_ranking(df, target_col)
        
        if ranking.empty:
            return []
        
        top_features = ranking.head(top_n)['feature'].tolist()
        logger.info(f"Selected top {len(top_features)} features")
        
        return top_features


def main():
    """Example usage of the FraudFeatureEngineer class."""
    
    # Initialize feature engineer
    engineer = FraudFeatureEngineer()
    
    # Example: Load IP country mapping
    # engineer.load_ip_country_mapping('data/IpAddress_to_Country.csv')
    
    # Example: Create features for e-commerce data
    # ecommerce_df = pd.read_csv('data/Fraud_Data.csv')
    # ecommerce_features = engineer.create_ecommerce_features(ecommerce_df)
    
    # Example: Create features for credit card data
    # creditcard_df = pd.read_csv('data/creditcard.csv')
    # creditcard_features = engineer.create_creditcard_features(creditcard_df)
    
    print("Fraud Feature Engineer initialized successfully!")


if __name__ == "__main__":
    main() 