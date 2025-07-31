"""
Data Preprocessing Module for Fraud Detection

This module handles all data preprocessing tasks including:
- Missing value handling
- Data cleaning
- Exploratory Data Analysis (EDA)
- Dataset merging for geolocation analysis
- Feature engineering
- Data transformation and scaling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDataProcessor:
    """
    Comprehensive data processor for fraud detection datasets.
    
    Handles e-commerce fraud data (Fraud_Data.csv) and credit card fraud data (creditcard.csv).
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data processor.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = None
        
    def load_data(self, fraud_data_path: str, ip_country_path: str = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load fraud detection datasets.
        
        Args:
            fraud_data_path: Path to Fraud_Data.csv
            ip_country_path: Path to IpAddress_to_Country.csv (optional)
            
        Returns:
            Tuple of (fraud_data, ip_country_data)
        """
        logger.info("Loading fraud detection datasets...")
        
        try:
            # Load main fraud data
            fraud_data = pd.read_csv(fraud_data_path)
            logger.info(f"Loaded fraud data: {fraud_data.shape}")
            
            # Load IP country mapping if provided
            ip_country_data = None
            if ip_country_path:
                ip_country_data = pd.read_csv(ip_country_path)
                logger.info(f"Loaded IP country data: {ip_country_data.shape}")
            
            return fraud_data, ip_country_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('auto', 'drop', 'impute')
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        
        df_clean = df.copy()
        
        # Check for missing values
        missing_summary = df_clean.isnull().sum()
        logger.info(f"Missing values summary:\n{missing_summary[missing_summary > 0]}")
        
        if strategy == 'auto':
            # Automatic strategy based on data type and missing percentage
            for column in df_clean.columns:
                missing_pct = df_clean[column].isnull().sum() / len(df_clean)
                
                if missing_pct > 0.5:
                    # Drop columns with more than 50% missing values
                    logger.info(f"Dropping column {column} with {missing_pct:.2%} missing values")
                    df_clean = df_clean.drop(columns=[column])
                elif missing_pct > 0:
                    # Impute based on data type
                    if df_clean[column].dtype in ['object', 'category']:
                        # Categorical: mode
                        mode_value = df_clean[column].mode()[0]
                        df_clean[column] = df_clean[column].fillna(mode_value)
                        logger.info(f"Imputed categorical column {column} with mode: {mode_value}")
                    else:
                        # Numerical: median
                        median_value = df_clean[column].median()
                        df_clean[column] = df_clean[column].fillna(median_value)
                        logger.info(f"Imputed numerical column {column} with median: {median_value}")
        
        elif strategy == 'drop':
            # Drop rows with any missing values
            df_clean = df_clean.dropna()
            logger.info(f"Dropped rows with missing values. New shape: {df_clean.shape}")
        
        elif strategy == 'impute':
            # Impute all missing values
            for column in df_clean.columns:
                if df_clean[column].isnull().sum() > 0:
                    if df_clean[column].dtype in ['object', 'category']:
                        df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
                    else:
                        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
        
        return df_clean
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by removing duplicates and correcting data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning dataset...")
        
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Convert timestamps to datetime
        timestamp_columns = ['signup_time', 'purchase_time']
        for col in timestamp_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                logger.info(f"Converted {col} to datetime")
        
        # Convert IP addresses to integer format
        if 'ip_address' in df_clean.columns:
            df_clean['ip_address_int'] = df_clean['ip_address'].apply(self._ip_to_int)
            logger.info("Converted IP addresses to integer format")
        
        # Ensure numerical columns are numeric
        numerical_columns = ['age', 'purchase_value']
        for col in numerical_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                logger.info(f"Converted {col} to numeric")
        
        return df_clean
    
    def _ip_to_int(self, ip_address: str) -> int:
        """
        Convert IP address string to integer.
        
        Args:
            ip_address: IP address string
            
        Returns:
            Integer representation of IP address
        """
        try:
            if pd.isna(ip_address):
                return 0
            
            parts = ip_address.split('.')
            if len(parts) != 4:
                return 0
            
            return sum(int(part) << (24 - 8 * i) for i, part in enumerate(parts))
        except:
            return 0
    
    def merge_datasets(self, fraud_df: pd.DataFrame, ip_country_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge fraud data with IP country mapping for geolocation analysis.
        
        Args:
            fraud_df: Fraud data DataFrame
            ip_country_df: IP country mapping DataFrame
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging datasets for geolocation analysis...")
        
        # Ensure IP addresses are in the same format
        if 'ip_address_int' not in fraud_df.columns:
            fraud_df['ip_address_int'] = fraud_df['ip_address'].apply(self._ip_to_int)
        
        if 'ip_address_int' not in ip_country_df.columns:
            ip_country_df['ip_address_int'] = ip_country_df['ip_address'].apply(self._ip_to_int)
        
        # Merge datasets
        merged_df = fraud_df.merge(
            ip_country_df[['ip_address_int', 'country']], 
            on='ip_address_int', 
            how='left'
        )
        
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        logger.info(f"Countries found: {merged_df['country'].nunique()}")
        
        return merged_df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features for fraud detection.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time features
        """
        logger.info("Creating time-based features...")
        
        df_features = df.copy()
        
        # Ensure timestamps are datetime
        if 'purchase_time' in df_features.columns:
            df_features['purchase_time'] = pd.to_datetime(df_features['purchase_time'])
            
            # Hour of day
            df_features['hour_of_day'] = df_features['purchase_time'].dt.hour
            
            # Day of week
            df_features['day_of_week'] = df_features['purchase_time'].dt.dayofweek
            
            # Month
            df_features['month'] = df_features['purchase_time'].dt.month
            
            # Day of month
            df_features['day_of_month'] = df_features['purchase_time'].dt.day
            
            # Weekend flag
            df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6])
            
            # Night time flag (10 PM - 6 AM)
            df_features['is_night'] = (df_features['hour_of_day'] >= 22) | (df_features['hour_of_day'] <= 6)
        
        # Time since signup
        if 'signup_time' in df_features.columns and 'purchase_time' in df_features.columns:
            df_features['signup_time'] = pd.to_datetime(df_features['signup_time'])
            df_features['time_since_signup'] = (
                df_features['purchase_time'] - df_features['signup_time']
            ).dt.total_seconds() / 3600  # Convert to hours
            
            # Quick purchase flag (within 1 hour of signup)
            df_features['quick_purchase'] = df_features['time_since_signup'] < 1
        
        logger.info("Time-based features created successfully")
        return df_features
    
    def create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create transaction frequency and velocity features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with transaction features
        """
        logger.info("Creating transaction features...")
        
        df_features = df.copy()
        
        # Transaction value features
        if 'purchase_value' in df_features.columns:
            # Log transformation
            df_features['log_purchase_value'] = np.log1p(df_features['purchase_value'])
            
            # High value transaction flag (top 5%)
            high_value_threshold = df_features['purchase_value'].quantile(0.95)
            df_features['is_high_value'] = df_features['purchase_value'] > high_value_threshold
            
            # Low value transaction flag (bottom 5%)
            low_value_threshold = df_features['purchase_value'].quantile(0.05)
            df_features['is_low_value'] = df_features['purchase_value'] < low_value_threshold
        
        # User-based transaction features
        if 'user_id' in df_features.columns:
            # Transaction count per user
            user_transaction_counts = df_features['user_id'].value_counts()
            df_features['user_transaction_count'] = df_features['user_id'].map(user_transaction_counts)
            
            # First time user flag
            df_features['is_first_time_user'] = df_features['user_transaction_count'] == 1
        
        # Device-based features
        if 'device_id' in df_features.columns:
            # Device transaction count
            device_transaction_counts = df_features['device_id'].value_counts()
            df_features['device_transaction_count'] = df_features['device_id'].map(device_transaction_counts)
        
        logger.info("Transaction features created successfully")
        return df_features
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical columns to encode
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        if categorical_columns is None:
            # Auto-detect categorical columns
            categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for column in categorical_columns:
            if column in df_encoded.columns:
                # Create label encoder
                le = LabelEncoder()
                df_encoded[f'{column}_encoded'] = le.fit_transform(df_encoded[column].astype(str))
                
                # Store encoder for later use
                self.label_encoders[column] = le
                
                logger.info(f"Encoded column: {column}")
        
        return df_encoded
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, 
                             method: str = 'smote', test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using various sampling techniques.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Sampling method ('smote', 'undersample', 'smoteenn')
            test_size: Proportion of test set
            
        Returns:
            Tuple of (X_train_resampled, y_train_resampled, X_test, y_test)
        """
        logger.info(f"Handling class imbalance using {method}...")
        
        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Analyze class distribution
        logger.info(f"Original class distribution:\n{y.value_counts()}")
        logger.info(f"Training set class distribution:\n{y_train.value_counts()}")
        
        if method == 'smote':
            # SMOTE oversampling
            smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
        elif method == 'undersample':
            # Random undersampling
            undersampler = RandomUnderSampler(random_state=self.random_state)
            X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
            
        elif method == 'smoteenn':
            # SMOTE + ENN (Edited Nearest Neighbors)
            smoteenn = SMOTEENN(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train, y_train)
            
        else:
            logger.warning(f"Unknown method {method}. Using original data.")
            X_train_resampled, y_train_resampled = X_train, y_train
        
        logger.info(f"Resampled training set class distribution:\n{pd.Series(y_train_resampled).value_counts()}")
        
        return X_train_resampled, y_train_resampled, X_test, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using specified method.
        
        Args:
            X_train: Training features
            X_test: Test features
            method: Scaling method ('standard', 'minmax')
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        logger.info(f"Scaling features using {method} scaler...")
        
        # Select numerical columns
        numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            logger.warning(f"Unknown scaling method {method}. Using StandardScaler.")
            scaler = StandardScaler()
        
        # Fit scaler on training data
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])
        
        # Store scaler for later use
        self.scalers[method] = scaler
        
        logger.info(f"Scaled {len(numerical_columns)} numerical features")
        
        return X_train_scaled, X_test_scaled
    
    def perform_eda(self, df: pd.DataFrame, target_column: str = 'class') -> Dict:
        """
        Perform Exploratory Data Analysis.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            
        Returns:
            Dictionary containing EDA results
        """
        logger.info("Performing Exploratory Data Analysis...")
        
        eda_results = {}
        
        # Basic information
        eda_results['shape'] = df.shape
        eda_results['columns'] = df.columns.tolist()
        eda_results['dtypes'] = df.dtypes.to_dict()
        eda_results['missing_values'] = df.isnull().sum().to_dict()
        
        # Target distribution
        if target_column in df.columns:
            eda_results['target_distribution'] = df[target_column].value_counts().to_dict()
            eda_results['target_balance'] = df[target_column].value_counts(normalize=True).to_dict()
        
        # Numerical features summary
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_columns:
            eda_results['numerical_summary'] = df[numerical_columns].describe().to_dict()
        
        # Categorical features summary
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_columns:
            eda_results['categorical_summary'] = {}
            for col in categorical_columns:
                eda_results['categorical_summary'][col] = df[col].value_counts().to_dict()
        
        logger.info("EDA completed successfully")
        return eda_results
    
    def create_eda_plots(self, df: pd.DataFrame, target_column: str = 'class', 
                        save_path: str = None) -> None:
        """
        Create EDA visualization plots.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            save_path: Path to save plots (optional)
        """
        logger.info("Creating EDA plots...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fraud Detection EDA', fontsize=16, fontweight='bold')
        
        # 1. Target distribution
        if target_column in df.columns:
            target_counts = df[target_column].value_counts()
            axes[0, 0].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Target Distribution')
        
        # 2. Purchase value distribution
        if 'purchase_value' in df.columns:
            axes[0, 1].hist(df['purchase_value'], bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Purchase Value Distribution')
            axes[0, 1].set_xlabel('Purchase Value')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Hour of day distribution
        if 'hour_of_day' in df.columns:
            hour_counts = df['hour_of_day'].value_counts().sort_index()
            axes[1, 0].bar(hour_counts.index, hour_counts.values)
            axes[1, 0].set_title('Transaction Distribution by Hour')
            axes[1, 0].set_xlabel('Hour of Day')
            axes[1, 0].set_ylabel('Number of Transactions')
        
        # 4. Correlation heatmap (numerical features only)
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_columns) > 1:
            correlation_matrix = df[numerical_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1, 1], fmt='.2f')
            axes[1, 1].set_title('Correlation Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"EDA plots saved to {save_path}")
        
        plt.show()
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get a summary of preprocessing steps performed.
        
        Returns:
            Dictionary with preprocessing summary
        """
        summary = {
            'scalers_used': list(self.scalers.keys()),
            'label_encoders_used': list(self.label_encoders.keys()),
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        return summary


def main():
    """Example usage of the FraudDataProcessor class."""
    
    # Initialize processor
    processor = FraudDataProcessor()
    
    # Example workflow
    print("Fraud Data Processor initialized successfully!")
    print("Use this class to:")
    print("1. Load and clean fraud detection datasets")
    print("2. Handle missing values and duplicates")
    print("3. Perform EDA and create visualizations")
    print("4. Merge datasets for geolocation analysis")
    print("5. Create time-based and transaction features")
    print("6. Handle class imbalance")
    print("7. Scale and encode features")


if __name__ == "__main__":
    main() 