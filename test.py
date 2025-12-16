#!/usr/bin/env python3
"""Test script to verify polymesh_ai installation."""

import sys

def test_basic_import():
    """Test basic package import."""
    print("Test 1: Basic import...")
    try:
        import polymesh_ai
        print("✓ polymesh_ai imported successfully")
        print(f"  Version: {polymesh_ai.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import polymesh_ai: {e}")
        return False

def test_library_info():
    """Test getting library info."""
    print("\nTest 2: Library info...")
    try:
        import polymesh_ai
        info = polymesh_ai.get_library_info()
        print("✓ Library info retrieved")
        print(f"  Components available: {sum(info['components'].values())}/4")
        return True
    except Exception as e:
        print(f"✗ Failed to get library info: {e}")
        return False

def test_mesh_generation():
    """Test mesh generation."""
    print("\nTest 3: Mesh generation...")
    try:
        import polymesh_ai
        mesh = polymesh_ai.generate_sample_mesh('cube', size=1.0)
        print(f"✓ Generated cube mesh")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        return True
    except Exception as e:
        print(f"✗ Failed to generate mesh: {e}")
        return False

def test_tokenization():
    """Test mesh tokenization."""
    print("\nTest 4: Mesh tokenization...")
    try:
        import polymesh_ai
        mesh = polymesh_ai.generate_sample_mesh('sphere', radius=1.0, subdivisions=1)
        mesh.compute_vertex_normals()
        
        tokenizer = polymesh_ai.create_vertex_tokenizer(include_normals=True)
        tokens = tokenizer.tokenize(mesh)
        
        print(f"✓ Tokenized mesh")
        print(f"  Tokens generated: {len(tokens)}")
        print(f"  Token type: {tokens[0].token_type if tokens else 'N/A'}")
        return True
    except Exception as e:
        print(f"✗ Failed to tokenize mesh: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation."""
    print("\nTest 5: Model creation...")
    try:
        import polymesh_ai
        import torch
        
        model = polymesh_ai.create_mesh_transformer(
            feature_dim=6,
            num_classes=10,
            task='classification'
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Created model")
        print(f"  Parameters: {param_count:,}")
        print(f"  Device: {next(model.parameters()).device}")
        return True
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test forward pass through model."""
    print("\nTest 6: Forward pass...")
    try:
        import polymesh_ai
        import torch
        
        # Generate mesh and tokenize
        mesh = polymesh_ai.generate_sample_mesh('sphere', radius=1.0, subdivisions=1)
        mesh.compute_vertex_normals()
        
        tokenizer = polymesh_ai.create_vertex_tokenizer(include_normals=True)
        tokens = tokenizer.tokenize(mesh)
        
        # Create model
        model = polymesh_ai.create_mesh_transformer(
            feature_dim=6,
            num_classes=3,
            task='classification'
        )
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            output = model(tokens, task='classification')
        
        print(f"✓ Forward pass successful")
        print(f"  Input tokens: {len(tokens)}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output values: {output[0][:3].tolist()}")
        return True
    except Exception as e:
        print(f"✗ Failed forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("POLYMESH AI INSTALLATION TEST")
    print("=" * 60)
    
    tests = [
        test_basic_import,
        test_library_info,
        test_mesh_generation,
        test_tokenization,
        test_model_creation,
        test_forward_pass,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)
        print()
    
    print("=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n🎉 All tests passed! Installation is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())