from setuptools import setup, find_packages
import os

# Read README if it exists, otherwise use a default description
readme_path = "README.md"
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = """
# Polymesh AI

A library for mesh processing with transformers.

## Features
- 3D mesh loading and manipulation
- Mesh tokenization strategies (vertex, face, patch-based)
- Transformer models for mesh processing
- Advanced attention mechanisms (geometric, graph, multi-scale)
- Training pipelines for mesh classification and reconstruction

## Installation
```bash
pip install polymesh-ai
```

## Quick Start
```python
import polymesh_ai

# Generate a sample mesh
mesh = polymesh_ai.generate_sample_mesh('sphere', radius=1.0, subdivisions=2)
mesh.compute_vertex_normals()

# Tokenize the mesh
tokenizer = polymesh_ai.create_vertex_tokenizer(include_normals=True)
tokens = tokenizer.tokenize(mesh)

# Create a transformer model
model = polymesh_ai.create_mesh_transformer(feature_dim=6, num_classes=10)
```
"""

setup(
    name="polymesh-ai",
    version="0.3.4",
    author="Matias Nielsen",
    author_email="your.email@example.com",
    description="A library for mesh processing with transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/polymesh-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "training": [
            "wandb>=0.12.0",
        ],
    },
)