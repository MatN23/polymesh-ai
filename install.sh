#!/bin/bash

echo "======================================"
echo "Polymesh AI Installation Script"
echo "======================================"
echo ""

# Step 1: Find the package
echo "Step 1: Finding your package..."
echo "Current directory: $(pwd)"
echo ""

# Check if we're in the right place
if [ -d "polymesh_ai" ]; then
    echo "✓ Found polymesh_ai directory!"
elif [ -d "polymesh-ai/polymesh_ai" ]; then
    echo "✓ Found polymesh_ai in polymesh-ai subdirectory!"
    cd polymesh-ai
else
    echo "✗ Cannot find polymesh_ai directory."
    echo "Looking for it..."
    find . -name "polymesh_ai" -type d 2>/dev/null | head -5
    echo ""
    echo "Please cd to the directory that contains 'polymesh_ai' folder"
    exit 1
fi

echo ""
echo "Step 2: Checking package contents..."
if [ -f "polymesh_ai/__init__.py" ]; then
    echo "✓ Found __init__.py"
else
    echo "✗ Missing __init__.py"
    exit 1
fi

echo ""
echo "Step 3: Creating pyproject.toml..."
cat > pyproject.toml << 'EOFCONFIG'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "polymesh-ai"
version = "0.3.4"
authors = [
    {name = "Matias Nielsen"}
]
description = "A library for mesh processing with transformers"
requires-python = ">=3.8"
dependencies = [
    "torch>=1.12.0",
    "numpy>=1.21.0",
    "tqdm>=4.62.0",
]

[tool.setuptools]
packages = ["polymesh_ai"]
EOFCONFIG

echo "✓ Created pyproject.toml"

echo ""
echo "Step 4: Installing package..."
pip install -e . --user

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Installation successful!"
    echo ""
    echo "Step 5: Testing installation..."
    python3 << 'EOFPYTHON'
import sys
try:
    import polymesh_ai
    print("✓ polymesh_ai imported successfully!")
    print(f"  Version: {polymesh_ai.__version__}")
    
    # Quick test
    mesh = polymesh_ai.generate_sample_mesh('cube', size=1.0)
    print(f"✓ Generated test mesh with {len(mesh.vertices)} vertices")
    
    print("\n🎉 Installation successful! You can now use polymesh_ai")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOFPYTHON
else
    echo "✗ Installation failed"
    exit 1
fi

echo ""
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""
echo "To use the library, run:"
echo "  python3"
echo "  >>> import polymesh_ai"
echo "  >>> polymesh_ai.quick_start_example()"
echo ""