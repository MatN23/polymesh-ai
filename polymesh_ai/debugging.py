# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import sys
import os
import subprocess
import importlib.util
from pathlib import Path

def run_command(cmd):
    """Run a command and return its output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def check_python_environment():
    """Check Python environment details"""
    print("=" * 60)
    print("PYTHON ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"Virtual environment: {sys.prefix}")
    else:
        print("Virtual environment: Not detected")
    
    print()

def check_package_installation():
    """Check if polymesh_ai is properly installed"""
    print("PACKAGE INSTALLATION CHECK")
    print("-" * 40)
    
    # Method 1: Try direct import
    try:
        import polymesh_ai
        print("✅ Direct import successful")
        print(f"   Package location: {polymesh_ai.__file__}")
        print(f"   Package version: {getattr(polymesh_ai, '__version__', 'Unknown')}")
    except ImportError as e:
        print(f"❌ Direct import failed: {e}")
    except Exception as e:
        print(f"❌ Direct import error: {e}")
    
    # Method 2: Check with pip list
    stdout, stderr, code = run_command("pip list | grep polymesh")
    if code == 0 and stdout:
        print(f"✅ Found in pip list: {stdout}")
    else:
        print("❌ Not found in pip list")
    
    # Method 3: Check with pip show
    stdout, stderr, code = run_command("pip show polymesh-ai")
    if code == 0:
        print("✅ Package details from pip show:")
        for line in stdout.split('\n')[:8]:  # Show first 8 lines
            print(f"   {line}")
    else:
        print("❌ pip show polymesh-ai failed")
        print(f"   Error: {stderr}")
    
    print()

def check_site_packages():
    """Check site-packages directory"""
    print("SITE-PACKAGES CHECK")
    print("-" * 40)
    
    import site
    site_packages = site.getsitepackages()
    
    for sp in site_packages:
        sp_path = Path(sp)
        if sp_path.exists():
            # Look for polymesh_ai
            polymesh_dirs = list(sp_path.glob("polymesh_ai*"))
            if polymesh_dirs:
                print(f"✅ Found in {sp}:")
                for d in polymesh_dirs:
                    print(f"   {d}")
            else:
                print(f"❌ Not found in {sp}")
        else:
            print(f"❌ Site-packages path doesn't exist: {sp}")
    
    print()

def check_local_development():
    """Check if there's a local development setup"""
    print("LOCAL DEVELOPMENT CHECK")
    print("-" * 40)
    
    # Check current directory
    cwd = Path.cwd()
    print(f"Current directory: {cwd}")
    
    # Look for setup.py, pyproject.toml, or polymesh_ai directory
    setup_files = ['setup.py', 'pyproject.toml', 'setup.cfg']
    found_setup = False
    
    for setup_file in setup_files:
        if (cwd / setup_file).exists():
            print(f"✅ Found {setup_file}")
            found_setup = True
    
    if not found_setup:
        print("❌ No setup files found in current directory")
    
    # Check for polymesh_ai source directory
    polymesh_src = cwd / "polymesh_ai"
    if polymesh_src.exists():
        print(f"✅ Found source directory: {polymesh_src}")
        init_file = polymesh_src / "__init__.py"
        if init_file.exists():
            print(f"✅ Found __init__.py: {init_file}")
        else:
            print(f"❌ Missing __init__.py in {polymesh_src}")
    else:
        print("❌ No polymesh_ai source directory found")
    
    print()

def suggest_fixes():
    """Suggest potential fixes"""
    print("POTENTIAL FIXES")
    print("-" * 40)
    
    print("1. Reinstall the package:")
    print("   pip uninstall polymesh-ai")
    print("   pip install polymesh-ai==0.3.4")
    print()
    
    print("2. Install in development mode (if you have source code):")
    print("   pip install -e .")
    print()
    
    print("3. Check virtual environment activation:")
    print("   conda activate pymesh3d  # or your environment name")
    print()
    
    print("4. Clear Python cache and reinstall:")
    print("   find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true")
    print("   pip uninstall polymesh-ai")
    print("   pip install polymesh-ai==0.3.4")
    print()
    
    print("5. Install from local wheel (if available):")
    print("   pip install dist/polymesh_ai-0.3.4-py3-none-any.whl --force-reinstall")
    print()

def run_diagnostics():
    """Run all diagnostic checks"""
    check_python_environment()
    check_package_installation()
    check_site_packages()
    check_local_development()
    suggest_fixes()

if __name__ == "__main__":
    print("Polymesh AI Import Diagnostics")
    print("=" * 60)
    run_diagnostics()
    
    print("\nDIAGNOSTICS COMPLETE")
    print("=" * 60)
    
    # Final test
    print("\nFINAL IMPORT TEST:")
    try:
        import polymesh_ai
        print("✅ SUCCESS: polymesh_ai imported successfully!")
        
        # Test a simple operation
        try:
            info = polymesh_ai.get_library_info()
            print(f"✅ Library info retrieved: {info['version']}")
        except Exception as e:
            print(f"⚠️  Import successful but function call failed: {e}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")