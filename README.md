# Fraud Detection Project - Adey Innovations Inc.

A comprehensive machine learning solution for fraud detection in e-commerce and banking transactions, developed for Adey Innovations Inc.

## 🎯 Project Overview

This project addresses the critical challenge of fraud detection in fintech applications, balancing **security** (catching actual fraud) with **user experience** (minimizing false alarms for legitimate customers).

### Business Context
- **Company**: Adey Innovations Inc. (Fintech specializing in e-commerce and banking solutions)
- **Problem**: Fraudulent transactions in both e-commerce and banking sectors
- **Goal**: Develop ML models that accurately detect fraud while maintaining good user experience

### Key Challenges
- **Class Imbalance**: Fraudulent transactions are very rare (<1% in credit card data)
- **Real-time Detection**: Must be fast enough for instant transaction blocking
- **False Positive vs False Negative Trade-off**: Security vs convenience

## 📊 Datasets

### 1. E-commerce Fraud Data (`Fraud_Data.csv`)
- **Goal**: Detect fraudulent online purchases
- **Features**: User info, transaction info, location info, device/browser data
- **Target**: `class` (1 = fraud, 0 = legitimate)

### 2. IP Address to Country Mapping (`IpAddress_to_Country.csv`)
- **Goal**: Map IP addresses to geographical locations
- **Use**: Geolocation-based fraud features

### 3. Credit Card Fraud Data (`creditcard.csv`)
- **Goal**: Detect fraudulent credit card transactions
- **Features**: Time, Amount, V1-V28 (PCA features)
- **Target**: `Class` (1 = fraud, 0 = legitimate)

## 🏗️ Project Structure

```
fraud-detection/
├── data/                          # Data storage
│   ├── raw/                       # Original datasets
│   ├── processed/                 # Preprocessed data
│   └── external/                  # External data sources
├── notebooks/                     # Jupyter notebooks
│   └── 01_data_analysis_preprocessing.ipynb  # Task 1 notebook
├── src/                          # Source code
│   ├── config/                   # Configuration
│   │   └── config.py             # Project settings
│   ├── data_preprocessing/       # Data preprocessing
│   │   └── data_processor.py     # Main preprocessing module
│   ├── features/                 # Feature engineering
│   │   └── feature_engineering.py # Feature creation
│   └── utils/                    # Utilities
│       └── data_validation.py    # Data validation
├── fastapi_app/                  # FastAPI web application
├── tests/                        # Unit and integration tests
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Containerization
├── .gitignore                    # Git exclusions
├── README.md                     # Project documentation
└── LICENSE                       # MIT License
```

## 🚀 Task 1: Data Analysis and Preprocessing

### ✅ Completed Implementation

#### 1. **Data Loading and Initial Exploration**
- Comprehensive data loading with error handling
- Initial data structure analysis
- Sample data generation for demonstration

#### 2. **Missing Value Handling**
- **Automatic Strategy**: Drop columns with >50% missing values, impute others
- **Smart Imputation**: Mode for categorical, median for numerical
- **Detailed Reporting**: Missing value analysis and visualization

#### 3. **Data Cleaning**
- **Duplicate Removal**: Automatic detection and removal
- **Data Type Conversion**: Timestamps, IP addresses, numerical columns
- **IP Address Conversion**: String to integer format for efficient processing

#### 4. **Exploratory Data Analysis (EDA)**
- **Univariate Analysis**: Distribution plots, summary statistics
- **Bivariate Analysis**: Correlation analysis, target relationships
- **Visualization**: Comprehensive EDA plots with business insights
- **Class Imbalance Analysis**: Detailed fraud vs legitimate transaction analysis

#### 5. **Dataset Merging for Geolocation Analysis**
- **IP Address Mapping**: Convert and merge with country data
- **Geolocation Features**: Country-based fraud patterns
- **Consistency Validation**: Data overlap and quality checks

#### 6. **Feature Engineering**
- **Time-Based Features**:
  - `hour_of_day`: Transaction hour (0-23)
  - `day_of_week`: Day of week (0-6)
  - `time_since_signup`: Duration between signup and purchase
  - `is_weekend`: Weekend transaction flag
  - `is_night`: Night time transaction flag
  - `quick_purchase`: Purchase within 1 hour of signup

- **Transaction Features**:
  - `log_purchase_value`: Log-transformed purchase amount
  - `is_high_value`: High-value transaction flag (top 5%)
  - `is_low_value`: Low-value transaction flag (bottom 5%)
  - `user_transaction_count`: Number of transactions per user
  - `device_transaction_count`: Number of transactions per device

#### 7. **Data Transformation**
- **Class Imbalance Handling**:
  - **SMOTE**: Synthetic Minority Over-sampling Technique
  - **Random Undersampling**: Alternative for extreme imbalance
  - **SMOTEENN**: Combined oversampling and undersampling
  - **Justification**: SMOTE chosen for maintaining data distribution while balancing classes

- **Feature Scaling**:
  - **StandardScaler**: Z-score normalization (mean=0, std=1)
  - **MinMaxScaler**: Range scaling (0-1)
  - **RobustScaler**: Outlier-resistant scaling

- **Categorical Encoding**:
  - **LabelEncoder**: Integer encoding for categorical variables
  - **One-Hot Encoding**: Binary encoding for nominal variables
  - **Target Encoding**: Mean encoding for high-cardinality features

## 🔧 Key Components

### 1. **FraudDataProcessor** (`src/data_preprocessing/data_processor.py`)
Comprehensive data processing class with methods for:
- Missing value handling with multiple strategies
- Data cleaning and validation
- EDA and visualization
- Feature engineering
- Class imbalance handling
- Feature scaling and encoding

### 2. **FraudFeatureEngineer** (`src/features/feature_engineering.py`)
Advanced feature engineering with:
- Time-based fraud patterns
- Transaction velocity features
- User behavior analysis
- Location-based features
- PCA statistical features (for credit card data)

### 3. **DataValidator** (`src/utils/data_validation.py`)
Data quality assurance with:
- Dataset structure validation
- Data type checking
- Quality scoring
- Consistency validation
- Comprehensive reporting

### 4. **Configuration Management** (`src/config/config.py`)
Centralized project configuration:
- Path management
- Hyperparameter settings
- Model configurations
- Evaluation metrics
- Environment settings

## 📈 Key Insights from Task 1

### Class Imbalance Analysis
- **E-commerce Data**: ~5% fraud rate (moderate imbalance)
- **Credit Card Data**: <1% fraud rate (severe imbalance)
- **Solution**: SMOTE oversampling for balanced training

### Feature Importance
- **Time-based features**: Strong correlation with fraud patterns
- **Transaction velocity**: Key indicator of suspicious activity
- **Geolocation**: Unusual location patterns signal fraud
- **Device/browser patterns**: Frequent changes indicate account takeover

### Data Quality Metrics
- **Missing Values**: <5% in most features
- **Data Consistency**: High overlap between datasets
- **Feature Quality**: Strong correlation with target variables

## 🎯 Business Impact

### Security Improvements
- **Early Fraud Detection**: Time-based features catch fraud quickly
- **Geolocation Monitoring**: Unusual location patterns flagged
- **Device Tracking**: Suspicious device changes detected

### User Experience
- **Reduced False Positives**: Balanced model training
- **Fast Processing**: Optimized feature engineering
- **Scalable Solution**: Containerized deployment ready

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd fraud-detection

# Run the preprocessing notebook
jupyter notebook notebooks/01_data_analysis_preprocessing.ipynb

# Or run the preprocessing script
python src/data_preprocessing/data_processor.py
```

### Docker Setup
```bash
# Start all services
docker-compose up -d

# Access services
# - FastAPI: http://localhost:8000
# - Jupyter: http://localhost:8888
# - MLflow: http://localhost:5000
```

## 📊 Evaluation Metrics

### Fraud-Specific Metrics
- **Precision-Recall AUC**: Better for imbalanced data than ROC-AUC
- **F1-Score**: Balance between precision and recall
- **Recall (Sensitivity)**: Important to catch fraud
- **Specificity**: Minimize false alarms

### Business Metrics
- **False Positive Rate**: Impact on user experience
- **False Negative Rate**: Financial loss from missed fraud
- **Processing Time**: Real-time detection requirements

## 🔮 Next Steps

### Task 2: Model Building and Evaluation
- Implement multiple ML algorithms (Random Forest, XGBoost, LightGBM)
- Hyperparameter tuning with cross-validation
- Model evaluation using fraud-specific metrics
- Ensemble methods for improved performance

### Task 3: Model Explainability (XAI)
- SHAP (SHapley Additive exPlanations) implementation
- Feature importance analysis
- Individual prediction explanations
- Business stakeholder communication

### Task 4: Production Deployment
- FastAPI web service development
- Real-time prediction pipeline
- Model monitoring and retraining
- Performance optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or support, please contact the development team at Adey Innovations Inc.

---

**Note**: This project is designed for educational and research purposes. Always ensure compliance with data privacy regulations and security best practices when deploying in production environments.
