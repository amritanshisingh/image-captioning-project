"""
Local Testing Script for Image Captioning AI
Test your model and application before deploying to Hugging Face
"""

import os
import sys
import torch
import pickle
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
        print(f"✓ {description}: {filepath} ({size:.2f} MB)")
        return True
    else:
        print(f"✗ {description}: {filepath} NOT FOUND")
        return False

def test_model_loading():
    """Test if model can be loaded"""
    print("\n[Testing Model Loading]")
    try:
        # Load vocabulary
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        print(f"✓ Vocabulary loaded: {len(vocab)} words")
        
        # Load model checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('best_model.pth', map_location=device)
        
        print(f"✓ Model checkpoint loaded")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        print(f"  - Device: {device}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False

def test_dependencies():
    """Test if all required packages are installed"""
    print("\n[Testing Dependencies]")
    required_packages = {
        'flask': 'Flask',
        'torch': 'PyTorch',
        'torchvision': 'torchvision',
        'PIL': 'Pillow',
        'numpy': 'NumPy'
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            all_installed = False
    
    return all_installed

def test_file_structure():
    """Test if all required files exist"""
    print("\n[Testing File Structure]")
    
    files_to_check = {
        'app.py': 'Flask application',
        'best_model.pth': 'Model weights',
        'vocab.pkl': 'Vocabulary',
        'requirements.txt': 'Dependencies',
        'Dockerfile': 'Docker configuration',
        'README.md': 'Documentation',
        'templates/index.html': 'HTML template',
        'static/style.css': 'Stylesheet',
        'static/script.js': 'JavaScript'
    }
    
    all_exist = True
    for filepath, description in files_to_check.items():
        if not check_file(filepath, description):
            all_exist = False
    
    return all_exist

def test_app_imports():
    """Test if app.py can be imported"""
    print("\n[Testing Application Imports]")
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try importing app components
        from flask import Flask
        print("✓ Flask can be imported")
        
        import torch.nn as nn
        from torchvision import transforms, models
        print("✓ PyTorch modules can be imported")
        
        from PIL import Image
        print("✓ Pillow can be imported")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {str(e)}")
        return False

def estimate_memory():
    """Estimate memory requirements"""
    print("\n[Memory Estimation]")
    model_size = os.path.getsize('best_model.pth') / (1024 * 1024)
    vocab_size = os.path.getsize('vocab.pkl') / (1024 * 1024)
    
    estimated_ram = model_size * 2 + 500  # Model + overhead
    
    print(f"Model size: {model_size:.2f} MB")
    print(f"Vocab size: {vocab_size:.2f} MB")
    print(f"Estimated RAM needed: ~{estimated_ram:.0f} MB")
    
    if estimated_ram < 2000:
        print("✓ Should work on CPU Basic (2GB RAM)")
    elif estimated_ram < 4000:
        print("⚠ Recommended: CPU Upgrade (4GB RAM)")
    else:
        print("⚠ May need GPU instance")

def generate_summary_report():
    """Generate a summary report"""
    print("\n" + "="*60)
    print("DEPLOYMENT READINESS SUMMARY")
    print("="*60)
    
    checks = {
        "File Structure": test_file_structure(),
        "Dependencies": test_dependencies(),
        "Model Loading": test_model_loading(),
        "App Imports": test_app_imports()
    }
    
    estimate_memory()
    
    print("\n" + "="*60)
    print("Results:")
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {check_name}")
    
    all_passed = all(checks.values())
    
    print("="*60)
    if all_passed:
        print("\n🎉 All checks passed! Ready for deployment.")
        print("\nNext steps:")
        print("1. Run: python app.py")
        print("2. Test locally at: http://localhost:7860")
        print("3. Deploy to Hugging Face Spaces")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        print("See SETUP_GUIDE.md for help.")
    print()

def main():
    print("="*60)
    print("IMAGE CAPTIONING AI - LOCAL TEST SUITE")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("\n⚠ Error: app.py not found!")
        print("Please run this script from the deployment directory.")
        return
    
    generate_summary_report()

if __name__ == "__main__":
    main()