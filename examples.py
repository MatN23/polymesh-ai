#!/usr/bin/env python3
"""
Examples of using polymesh_ai library
Run this from any directory after installing polymesh-ai
"""

import polymesh_ai
import numpy as np

print("=" * 60)
print("POLYMESH AI USAGE EXAMPLES")
print("=" * 60)

# Example 1: Generate and manipulate meshes
print("\n1. Generate and Manipulate Meshes")
print("-" * 40)

# Create different mesh types
cube = polymesh_ai.generate_sample_mesh('cube', size=1.0)
sphere = polymesh_ai.generate_sample_mesh('sphere', radius=1.5, subdivisions=2)
cylinder = polymesh_ai.generate_sample_mesh('cylinder', radius=0.8, height=2.0)

print(f"Cube: {len(cube.vertices)} vertices, {len(cube.faces)} faces")
print(f"Sphere: {len(sphere.vertices)} vertices, {len(sphere.faces)} faces")
print(f"Cylinder: {len(cylinder.vertices)} vertices, {len(cylinder.faces)} faces")

# Compute normals
sphere.compute_vertex_normals()
print(f"✓ Computed normals for sphere")

# Normalize
sphere.normalize()
print(f"✓ Normalized sphere to unit scale")

# Example 2: Tokenize meshes
print("\n2. Tokenize Meshes")
print("-" * 40)

# Create tokenizer
tokenizer = polymesh_ai.create_vertex_tokenizer(
    include_normals=True,
    include_colors=False
)

# Tokenize the sphere
tokens = tokenizer.tokenize(sphere)
print(f"Generated {len(tokens)} tokens from sphere")
print(f"First token: {tokens[0]}")
print(f"Token features shape: {tokens[0].features.shape}")

# Example 3: Create transformer models
print("\n3. Create Transformer Models")
print("-" * 40)

# Simple classifier
classifier = polymesh_ai.create_mesh_transformer(
    feature_dim=6,  # 3D position + 3D normal
    num_classes=10,
    task='classification'
)

param_count = sum(p.numel() for p in classifier.parameters())
print(f"Created classifier with {param_count:,} parameters")

# Adaptive model
adaptive_model = polymesh_ai.create_mesh_transformer(
    feature_dim=6,
    num_classes=5,
    task='classification'
)
print(f"Created adaptive model")

# Example 4: Process a mesh through the model
print("\n4. Process Mesh Through Model")
print("-" * 40)

import torch

# Generate and prepare a test mesh
test_mesh = polymesh_ai.generate_sample_mesh('cube', size=1.0)
test_mesh.normalize().compute_vertex_normals()

# Tokenize
test_tokens = tokenizer.tokenize(test_mesh)
print(f"Input: {len(test_tokens)} tokens")

# Run through model (inference mode)
classifier.eval()
with torch.no_grad():
    output = classifier(test_tokens, task='classification')
    
print(f"Output shape: {output.shape}")
print(f"Predictions: {output[0][:3].tolist()}")

# Get predicted class
predicted_class = torch.argmax(output, dim=1).item()
print(f"Predicted class: {predicted_class}")

# Example 5: Load and save meshes
print("\n5. Load and Save Meshes")
print("-" * 40)

# Save a mesh to OBJ file
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    obj_path = os.path.join(tmpdir, "test_mesh.obj")
    
    # Save
    polymesh_ai.MeshLoader.save_obj(sphere, obj_path)
    print(f"✓ Saved mesh to {obj_path}")
    
    # Load it back
    loaded_mesh = polymesh_ai.load_mesh_obj(obj_path)
    print(f"✓ Loaded mesh: {len(loaded_mesh.vertices)} vertices")

# Example 6: Mesh operations
print("\n6. Mesh Operations")
print("-" * 40)

# Create a mesh
mesh = polymesh_ai.generate_sample_mesh('sphere', radius=1.0, subdivisions=1)

# Get properties
center = mesh.get_center()
scale = mesh.get_scale()
bbox_min, bbox_max = mesh.get_bounding_box()

print(f"Center: {center}")
print(f"Scale: {scale:.3f}")
print(f"Bounding box: {bbox_min} to {bbox_max}")

# Transform
mesh.translate([1.0, 0.0, 0.0])
mesh.scale(2.0)
print(f"✓ Translated and scaled")
print(f"New center: {mesh.get_center()}")

# Get adjacency
adjacency = mesh.get_adjacency_matrix()
print(f"Adjacency matrix shape: {adjacency.shape}")
print(f"Connected edges: {np.sum(adjacency) / 2:.0f}")

# Example 7: Using configurations
print("\n7. Using Pre-defined Configurations")
print("-" * 40)

# Get a config
config = polymesh_ai.get_config('classification')
print(f"Classification config:")
print(f"  Model dim: {config['d_model']}")
print(f"  Num layers: {config['num_layers']}")
print(f"  Learning rate: {config['learning_rate']}")

# Example 8: Library info
print("\n8. Library Information")
print("-" * 40)

info = polymesh_ai.get_library_info()
print(f"Version: {info['version']}")
print(f"Author: {info['author']}")
print(f"Components loaded:")
for component, loaded in info['components'].items():
    status = "✓" if loaded else "✗"
    print(f"  {status} {component}")

print("\n" + "=" * 60)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 60)

# Quick reference
print("\nQUICK REFERENCE:")
print("-" * 40)
print("Import: import polymesh_ai")
print("Generate mesh: polymesh_ai.generate_sample_mesh('sphere')")
print("Load mesh: polymesh_ai.load_mesh_obj('path/to/mesh.obj')")
print("Create tokenizer: polymesh_ai.create_vertex_tokenizer()")
print("Create model: polymesh_ai.create_mesh_transformer()")
print("Get config: polymesh_ai.get_config('classification')")
print("Check install: polymesh_ai.check_installation()")
print("Library info: polymesh_ai.get_library_info()")