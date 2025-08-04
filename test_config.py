#!/usr/bin/env python3
"""
Test script to validate chaamo-backend configuration
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        print("✅ Uvicorn imported successfully")
    except ImportError as e:
        print(f"❌ Uvicorn import failed: {e}")
        return False
    
    try:
        import playwright
        print("✅ Playwright imported successfully")
    except ImportError as e:
        print(f"❌ Playwright import failed: {e}")
        return False
    
    try:
        import supabase
        print("✅ Supabase imported successfully")
    except ImportError as e:
        print(f"❌ Supabase import failed: {e}")
        return False
    
    try:
        import beautifulsoup4
        print("✅ BeautifulSoup4 imported successfully")
    except ImportError as e:
        print(f"❌ BeautifulSoup4 import failed: {e}")
        return False
    
    return True

def test_app_creation():
    """Test if the FastAPI app can be created"""
    print("\n🔍 Testing FastAPI app creation...")
    
    try:
        from main import app
        print("✅ FastAPI app created successfully")
        return True
    except Exception as e:
        print(f"❌ FastAPI app creation failed: {e}")
        return False

def test_openapi_schema():
    """Test if OpenAPI schema can be generated"""
    print("\n🔍 Testing OpenAPI schema generation...")
    
    try:
        from main import app
        schema = app.openapi()
        print("✅ OpenAPI schema generated successfully")
        print(f"   📊 Schema version: {schema.get('openapi', 'unknown')}")
        print(f"   📝 Title: {schema.get('info', {}).get('title', 'unknown')}")
        print(f"   🔗 Paths: {len(schema.get('paths', {}))}")
        return True
    except Exception as e:
        print(f"❌ OpenAPI schema generation failed: {e}")
        return False

def test_environment_variables():
    """Test if required environment variables are set"""
    print("\n🔍 Testing environment variables...")
    
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    
    if supabase_url:
        print("✅ SUPABASE_URL is set")
    else:
        print("⚠️  SUPABASE_URL is not set (will be required for deployment)")
    
    if supabase_key:
        print("✅ SUPABASE_SERVICE_KEY is set")
    else:
        print("⚠️  SUPABASE_SERVICE_KEY is not set (will be required for deployment)")
    
    return True

def test_file_structure():
    """Test if all required files exist"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        "main.py",
        "requirements.txt",
        "Dockerfile",
        "fly.toml",
        "src/api/v1/endpoint.py",
        "src/api/v1/__init__.py",
        "src/handlers/ebay_search.py",
        "src/models/ebay.py",
        "src/utils/playwright.py",
        "src/utils/supabase.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🚀 Testing chaamo-backend configuration...\n")
    
    tests = [
        test_imports,
        test_app_creation,
        test_openapi_schema,
        test_environment_variables,
        test_file_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! The application is ready for deployment.")
        print("\n🚀 To deploy to fly.io:")
        print("   1. Set environment variables:")
        print("      export SUPABASE_URL='your_supabase_url'")
        print("      export SUPABASE_SERVICE_KEY='your_service_key'")
        print("   2. Run: ./deploy.sh")
        print("   3. Or run: fly deploy")
    else:
        print("❌ Some tests failed. Please fix the issues before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
