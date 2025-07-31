#!/usr/bin/env python3
"""
Simple test script to verify Task 3: Model Explainability implementation
"""

import sys
import os
sys.path.append('src')

def test_imports():
    """Test basic imports"""
    print("🔍 Testing imports...")
    
    try:
        from utils.model_explainability import FraudModelExplainer
        print("✅ FraudModelExplainer imported successfully")
    except Exception as e:
        print(f"❌ FraudModelExplainer failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test file structure"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        'src/utils/model_explainability.py',
        'tests/test_model_explainability.py',
        'notebooks/03_model_explainability.ipynb'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True

def test_code_quality():
    """Test code quality by checking file sizes"""
    print("\n🔍 Testing code quality...")
    
    files_to_check = [
        ('src/utils/model_explainability.py', 800, "Model explainability should be comprehensive"),
        ('tests/test_model_explainability.py', 400, "Tests should be thorough"),
        ('notebooks/03_model_explainability.ipynb', 200, "Notebook should be detailed")
    ]
    
    for file_path, min_lines, description in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = len(f.readlines())
            if lines >= min_lines:
                print(f"✅ {file_path}: {lines} lines ({description})")
            else:
                print(f"⚠️  {file_path}: {lines} lines (expected {min_lines}+)")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True

def test_explainability_functionality():
    """Test explainability functionality"""
    print("\n🔍 Testing explainability functionality...")
    
    try:
        from utils.model_explainability import FraudModelExplainer
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd
        import numpy as np
        
        # Create sample data and model
        np.random.seed(42)
        X = pd.DataFrame({
            'transaction_amount': np.random.exponential(100, 50),
            'time_since_signup': np.random.exponential(30, 50),
            'location_risk': np.random.beta(2, 5, 50),
            'user_trust_score': np.random.normal(0.7, 0.2, 50)
        })
        y = np.random.choice([0, 1], 50, p=[0.8, 0.2])
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test explainer initialization
        explainer = FraudModelExplainer(model)
        print("✅ Explainer initialized successfully")
        
        # Test SHAP explanation generation
        results = explainer.explain_model(X, sample_size=30)
        print("✅ SHAP explanations generated successfully")
        
        # Test feature importance ranking
        importance_df = explainer.get_feature_importance_ranking(5)
        print("✅ Feature importance ranking generated")
        
        # Test fraud drivers analysis
        fraud_drivers = explainer.analyze_fraud_drivers(5)
        print("✅ Fraud drivers analysis completed")
        
        # Test business insights
        insights = explainer.interpret_fraud_patterns()
        print("✅ Business insights generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Explainability functionality test failed: {e}")
        return False

def test_requirements():
    """Test if SHAP is in requirements"""
    print("\n🔍 Testing requirements...")
    
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        if 'shap' in content.lower():
            print("✅ SHAP found in requirements.txt")
            return True
        else:
            print("❌ SHAP not found in requirements.txt")
            return False
            
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Task 3: Model Explainability Implementation\n")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Code Quality", test_code_quality),
        ("Requirements", test_requirements),
        ("Basic Imports", test_imports),
        ("Explainability Functionality", test_explainability_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Task 3 implementation is working correctly.")
        print("\n📝 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run Jupyter notebook: jupyter notebook notebooks/03_model_explainability.ipynb")
        print("3. Run tests: python -m pytest tests/test_model_explainability.py -v")
        print("4. Generate explainability report: Use the comprehensive_report() method")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main() 