# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

__version__ = "0.3.4"
__author__ = "Matias Nielsen"

# Import core dependencies first
import sys
import warnings

# Check for required dependencies
_REQUIRED_DEPS = {
    'torch': 'PyTorch',
    'numpy': 'NumPy'
}

_MISSING_DEPS = []
for module, name in _REQUIRED_DEPS.items():
    try:
        __import__(module)
    except ImportError:
        _MISSING_DEPS.append(name)

if _MISSING_DEPS:
    warnings.warn(
        f"Missing required dependencies: {', '.join(_MISSING_DEPS)}. "
        f"Install with: pip install torch numpy"
    )

# Core mesh library components
try:
    from .mesh_library import (
        Vertex,
        Face, 
        Mesh,
        MeshGenerator,
        MeshLoader,
        MeshDataset
    )
    _MESH_LIBRARY_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Failed to import mesh_library: {e}")
    _MESH_LIBRARY_AVAILABLE = False

# Tokenization components
try:
    from .mesh_transformers import (
        MeshToken,
        VertexTokenizer,
        FaceTokenizer,
        PatchTokenizer,
        PositionalEncoding,
        GeometricSelfAttention,
        MeshTransformerLayer,
        MeshTransformer,
        AdaptiveMeshTransformer,
        create_mesh_classifier,
        create_mesh_autoencoder,
        create_adaptive_classifier
    )
    _MESH_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Failed to import mesh_transformers: {e}")
    _MESH_TRANSFORMERS_AVAILABLE = False

# Advanced attention mechanisms
try:
    from .mesh_attention import (
        GeometricAttention,
        GraphAttention,
        MultiScaleAttention,
        SparseAttention,
        MeshTransformerLayer as AdvancedMeshTransformerLayer,
        AdaptiveMeshTransformer as AdvancedAdaptiveMeshTransformer,
        MeshTransformerPreTrainer
    )
    _MESH_ATTENTION_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Failed to import mesh_attention: {e}")
    _MESH_ATTENTION_AVAILABLE = False

# Training pipeline components
try:
    from .mesh_training_pipeline import (
        MeshTransformerDataset,
        MeshTransformerTrainingPipeline,
        MeshAugmentation,
        MeshDatasetBuilder,
        train_mesh_classifier,
        train_mesh_autoencoder
    )
    _MESH_TRAINING_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Failed to import mesh_training_pipeline: {e}")
    _MESH_TRAINING_AVAILABLE = False

# Build __all__ dynamically based on what imported successfully
__all__ = [
    "__version__",
    "__author__",
]

if _MESH_LIBRARY_AVAILABLE:
    __all__.extend([
        "Vertex", "Face", "Mesh", "MeshGenerator", "MeshLoader", "MeshDataset"
    ])

if _MESH_TRANSFORMERS_AVAILABLE:
    __all__.extend([
        "MeshToken", "VertexTokenizer", "FaceTokenizer", "PatchTokenizer",
        "PositionalEncoding", "GeometricSelfAttention", "MeshTransformerLayer",
        "MeshTransformer", "AdaptiveMeshTransformer",
        "create_mesh_classifier", "create_mesh_autoencoder", "create_adaptive_classifier"
    ])

if _MESH_ATTENTION_AVAILABLE:
    __all__.extend([
        "GeometricAttention", "GraphAttention", "MultiScaleAttention",
        "SparseAttention", "MeshTransformerPreTrainer"
    ])

if _MESH_TRAINING_AVAILABLE:
    __all__.extend([
        "MeshTransformerDataset", "MeshTransformerTrainingPipeline",
        "MeshAugmentation", "MeshDatasetBuilder",
        "train_mesh_classifier", "train_mesh_autoencoder"
    ])

# Utility functions
def create_vertex_tokenizer(include_normals=True, include_colors=False, quantize=False):
    """Create a vertex tokenizer with common settings."""
    if not _MESH_TRANSFORMERS_AVAILABLE:
        raise ImportError("mesh_transformers module not available")
    
    return VertexTokenizer(
        include_normals=include_normals,
        include_colors=include_colors
    )

def create_mesh_transformer(feature_dim, num_classes=None, task='classification'):
    """Create a mesh transformer model with sensible defaults."""
    if not _MESH_TRANSFORMERS_AVAILABLE:
        raise ImportError("mesh_transformers module not available")
    
    if task == 'classification' and num_classes:
        return create_mesh_classifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            d_model=256,
            nhead=8,
            num_layers=4
        )
    else:
        return MeshTransformer(
            feature_dim=feature_dim,
            d_model=512,
            nhead=8,
            num_layers=6,
            num_classes=num_classes or 10
        )

def load_mesh_obj(filepath):
    """Quick utility to load a mesh from OBJ file."""
    if not _MESH_LIBRARY_AVAILABLE:
        raise ImportError("mesh_library module not available")
    
    return MeshLoader.load_obj(filepath)

def generate_sample_mesh(mesh_type='cube', **kwargs):
    """Generate a sample mesh for testing."""
    if not _MESH_LIBRARY_AVAILABLE:
        raise ImportError("mesh_library module not available")
    
    if mesh_type == 'cube':
        return MeshGenerator.cube(kwargs.get('size', 1.0))
    elif mesh_type == 'sphere':
        return MeshGenerator.sphere(
            radius=kwargs.get('radius', 1.0),
            subdivisions=kwargs.get('subdivisions', 2)
        )
    elif mesh_type == 'cylinder':
        return MeshGenerator.cylinder(
            radius=kwargs.get('radius', 1.0),
            height=kwargs.get('height', 2.0),
            segments=kwargs.get('segments', 16)
        )
    elif mesh_type == 'plane':
        return MeshGenerator.plane(
            width=kwargs.get('width', 2.0),
            height=kwargs.get('height', 2.0)
        )
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")

# Configuration presets
CLASSIFICATION_CONFIG = {
    'model_type': 'standard',
    'tokenizer_type': 'vertex',
    'feature_dim': 6,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'dim_feedforward': 1024,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'max_epochs': 100,
    'task_type': 'classification',
}

AUTOENCODER_CONFIG = {
    'model_type': 'standard',
    'tokenizer_type': 'vertex', 
    'feature_dim': 6,
    'd_model': 512,
    'nhead': 8,
    'num_layers': 8,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'learning_rate': 5e-5,
    'batch_size': 16,
    'max_epochs': 200,
    'task_type': 'reconstruction',
}

def get_config(config_name):
    """Get a predefined configuration."""
    configs = {
        'classification': CLASSIFICATION_CONFIG.copy(),
        'autoencoder': AUTOENCODER_CONFIG.copy(),
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]

# Library information
def get_library_info():
    """Get information about the mesh transformers library."""
    info = {
        'version': __version__,
        'author': __author__,
        'components': {
            'mesh_library': _MESH_LIBRARY_AVAILABLE,
            'mesh_transformers': _MESH_TRANSFORMERS_AVAILABLE,
            'mesh_attention': _MESH_ATTENTION_AVAILABLE,
            'mesh_training_pipeline': _MESH_TRAINING_AVAILABLE
        },
        'supported_tasks': [
            'Mesh classification',
            'Mesh reconstruction', 
            'Mesh generation',
        ],
    }
    return info

def print_library_info():
    """Print library information."""
    info = get_library_info()
    print(f"Polymesh AI Library v{info['version']}")
    print("=" * 50)
    print(f"Author: {info['author']}")
    print("\nComponents Status:")
    for component, available in info['components'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {component}")
    print("\nSupported Tasks:")
    for task in info['supported_tasks']:
        print(f"  • {task}")

def check_installation():
    """Check if the library is properly installed."""
    print("Checking Polymesh AI installation...")
    print("-" * 50)
    
    all_good = True
    
    # Check dependencies
    for module, name in _REQUIRED_DEPS.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            all_good = False
    
    # Check modules
    components = {
        'mesh_library': _MESH_LIBRARY_AVAILABLE,
        'mesh_transformers': _MESH_TRANSFORMERS_AVAILABLE,
        'mesh_attention': _MESH_ATTENTION_AVAILABLE,
        'mesh_training_pipeline': _MESH_TRAINING_AVAILABLE
    }
    
    for name, available in components.items():
        if available:
            print(f"✓ {name} loaded")
        else:
            print(f"✗ {name} failed to load")
            all_good = False
    
    print("-" * 50)
    if all_good:
        print("✓ Installation OK!")
    else:
        print("✗ Installation has issues")
        print("\nTry reinstalling: pip install --force-reinstall polymesh-ai")
    
    return all_good

# Quick start example
def quick_start_example():
    """Demonstrate basic library usage."""
    if not _MESH_LIBRARY_AVAILABLE or not _MESH_TRANSFORMERS_AVAILABLE:
        print("Required modules not available for quick start")
        return
    
    print("Polymesh AI Quick Start Example")
    print("=" * 40)
    
    # Generate sample mesh
    print("1. Generating sample sphere mesh...")
    mesh = generate_sample_mesh('sphere', radius=1.0, subdivisions=1)
    mesh.compute_vertex_normals()
    print(f"   Created mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Create tokenizer
    print("2. Creating vertex tokenizer...")
    tokenizer = create_vertex_tokenizer(include_normals=True)
    tokens = tokenizer.tokenize(mesh)
    print(f"   Generated {len(tokens)} tokens")
    
    # Create model
    print("3. Creating mesh transformer model...")
    model = create_mesh_transformer(feature_dim=6, num_classes=10, task='classification')
    
    import torch
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {param_count:,} parameters")
    
    print("\n✓ Quick start completed successfully!")
    return True

# Initialize
def initialize():
    """Initialize the polymesh_ai library."""
    try:
        check_installation()
        return True
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return False