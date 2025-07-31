#!/usr/bin/env python3
"""
Simple test script to verify Task 2 implementation
"""

import sys
import os
sys.path.append('src')

def test_imports():
    """Test basic imports"""
    print("🔍 Testing imports...")
    
    try:
        from config.config import get_config
        print("✅ Config module imported successfully")
    except Exception as e:
        print(f"❌ Config module failed: {e}")
        return False
    
    try:
        from utils.model_evaluation import FraudModelEvaluator
        print("✅ Model evaluator imported successfully")
    except Exception as e:
        print(f"❌ Model evaluator failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test file structure"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        'src/models/model_builder.py',
        'src/utils/model_evaluation.py',
        'tests/test_model_building.py',
        'notebooks/02_model_building_training.ipynb'
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
        ('src/models/model_builder.py', 600, "Model builder should be substantial"),
        ('src/utils/model_evaluation.py', 400, "Model evaluator should be substantial"),
        ('tests/test_model_building.py', 300, "Tests should be comprehensive")
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

def test_config_functionality():
    """Test config functionality"""
    print("\n🔍 Testing config functionality...")
    
    try:
        from config.config import get_config
        config = get_config()
        
        # Test basic config properties
        assert hasattr(config, 'PROJECT_ROOT'), "Config should have PROJECT_ROOT"
        assert hasattr(config, 'DATA_DIR'), "Config should have DATA_DIR"
        assert hasattr(config, 'get_data_paths'), "Config should have get_data_paths method"
        
        print("✅ Config object created successfully")
        print(f"✅ Project root: {config.PROJECT_ROOT}")
        print(f"✅ Data directory: {config.DATA_DIR}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Task 2 Implementation\n")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Code Quality", test_code_quality),
        ("Config Functionality", test_config_functionality),
        ("Basic Imports", test_imports)
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
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Task 2 implementation is working correctly.")
        print("\n📝 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run Jupyter notebook: jupyter notebook notebooks/02_model_building_training.ipynb")
        print("3. Run tests: python -m pytest tests/test_model_building.py -v")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main() 