# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import os
import json
import pickle
from abc import ABC, abstractmethod
import math

class Vertex:
    """Represents a single vertex in a 3D mesh"""
    
    def __init__(self, position: Union[List, np.ndarray], 
                 normal: Optional[Union[List, np.ndarray]] = None,
                 color: Optional[Union[List, np.ndarray]] = None,
                 uv: Optional[Union[List, np.ndarray]] = None,
                 vertex_id: Optional[int] = None):
        self.position = np.array(position, dtype=np.float32)
        self.normal = np.array(normal, dtype=np.float32) if normal is not None else None
        self.color = np.array(color, dtype=np.float32) if color is not None else None
        self.uv = np.array(uv, dtype=np.float32) if uv is not None else None
        self.vertex_id = vertex_id
        
    def __str__(self):
        return f"Vertex(pos={self.position}, normal={self.normal})"
    
    def __repr__(self):
        return self.__str__()

class Face:
    """Represents a face (polygon) in a 3D mesh"""
    
    def __init__(self, vertex_indices: List[int], 
                 normal: Optional[Union[List, np.ndarray]] = None,
                 face_id: Optional[int] = None):
        self.vertex_indices = list(vertex_indices)
        self.normal = np.array(normal, dtype=np.float32) if normal is not None else None
        self.face_id = face_id
        
    @property
    def size(self) -> int:
        """Number of vertices in this face"""
        return len(self.vertex_indices)
    
    def is_triangle(self) -> bool:
        return len(self.vertex_indices) == 3
    
    def is_quad(self) -> bool:
        return len(self.vertex_indices) == 4
    
    def __str__(self):
        return f"Face(vertices={self.vertex_indices}, normal={self.normal})"
    
    def __repr__(self):
        return self.__str__()

class Mesh:
    """Main 3D mesh class with vertices, faces, and utility methods"""
    
    def __init__(self, vertices: Optional[List[Vertex]] = None, 
                 faces: Optional[List[Face]] = None,
                 name: str = "Mesh"):
        self.vertices = vertices or []
        self.faces = faces or []
        self.name = name
        self._adjacency_matrix = None
        self._vertex_positions_cache = None
        self._face_normals_cache = None
        self._bbox_cache = None
        
    @property
    def vertex_positions(self) -> np.ndarray:
        """Get all vertex positions as numpy array [N, 3]"""
        if self._vertex_positions_cache is None:
            if not self.vertices:
                return np.empty((0, 3), dtype=np.float32)
            self._vertex_positions_cache = np.array([v.position for v in self.vertices], dtype=np.float32)
        return self._vertex_positions_cache
    
    @property
    def face_indices(self) -> List[List[int]]:
        """Get all face vertex indices"""
        return [face.vertex_indices for face in self.faces]
    
    def add_vertex(self, position: Union[List, np.ndarray], 
                   normal: Optional[Union[List, np.ndarray]] = None,
                   color: Optional[Union[List, np.ndarray]] = None) -> int:
        """Add a vertex and return its index"""
        vertex_id = len(self.vertices)
        vertex = Vertex(position, normal, color, vertex_id=vertex_id)
        self.vertices.append(vertex)
        self._invalidate_caches()
        return vertex_id
    
    def add_face(self, vertex_indices: List[int]) -> int:
        """Add a face and return its index"""
        if not all(0 <= idx < len(self.vertices) for idx in vertex_indices):
            raise ValueError("Face contains invalid vertex indices")
        
        face_id = len(self.faces)
        face = Face(vertex_indices, face_id=face_id)
        self.faces.append(face)
        self._invalidate_caches()
        return face_id
    
    def compute_vertex_normals(self) -> 'Mesh':
        """Compute vertex normals from face normals"""
        if not self.faces:
            return self
        
        # Initialize vertex normals to zero
        vertex_normals = np.zeros((len(self.vertices), 3), dtype=np.float32)
        vertex_counts = np.zeros(len(self.vertices), dtype=np.int32)
        
        # Compute face normals and accumulate to vertices
        for face in self.faces:
            if len(face.vertex_indices) >= 3:
                # Get face vertices
                v_indices = face.vertex_indices[:3]  # Use first 3 vertices for normal
                v0 = self.vertices[v_indices[0]].position
                v1 = self.vertices[v_indices[1]].position
                v2 = self.vertices[v_indices[2]].position
                
                # Compute face normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                face_normal = np.cross(edge1, edge2)
                face_normal_length = np.linalg.norm(face_normal)
                
                if face_normal_length > 1e-8:
                    face_normal = face_normal / face_normal_length
                    
                    # Accumulate to all face vertices
                    for v_idx in face.vertex_indices:
                        vertex_normals[v_idx] += face_normal
                        vertex_counts[v_idx] += 1
        
        # Normalize vertex normals
        for i in range(len(self.vertices)):
            if vertex_counts[i] > 0:
                vertex_normals[i] /= vertex_counts[i]
                # Normalize to unit length
                normal_length = np.linalg.norm(vertex_normals[i])
                if normal_length > 1e-8:
                    vertex_normals[i] /= normal_length
                self.vertices[i].normal = vertex_normals[i]
        
        return self
    
    def compute_face_normals(self) -> np.ndarray:
        """Compute face normals and cache them"""
        if self._face_normals_cache is not None:
            return self._face_normals_cache
        
        face_normals = []
        for face in self.faces:
            if len(face.vertex_indices) >= 3:
                v_indices = face.vertex_indices[:3]
                v0 = self.vertices[v_indices[0]].position
                v1 = self.vertices[v_indices[1]].position
                v2 = self.vertices[v_indices[2]].position
                
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal_length = np.linalg.norm(normal)
                
                if normal_length > 1e-8:
                    normal = normal / normal_length
                else:
                    normal = np.array([0, 0, 1], dtype=np.float32)
                
                face.normal = normal
                face_normals.append(normal)
            else:
                # Degenerate face
                normal = np.array([0, 0, 1], dtype=np.float32)
                face.normal = normal
                face_normals.append(normal)
        
        self._face_normals_cache = np.array(face_normals, dtype=np.float32)
        return self._face_normals_cache
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get mesh bounding box as (min_point, max_point)"""
        if self._bbox_cache is None:
            if not self.vertices:
                return np.zeros(3), np.zeros(3)
            
            positions = self.vertex_positions
            self._bbox_cache = (np.min(positions, axis=0), np.max(positions, axis=0))
        
        return self._bbox_cache
    
    def get_center(self) -> np.ndarray:
        """Get mesh center point"""
        min_point, max_point = self.get_bounding_box()
        return (min_point + max_point) / 2.0
    
    def get_scale(self) -> float:
        """Get mesh scale (maximum dimension)"""
        min_point, max_point = self.get_bounding_box()
        return np.max(max_point - min_point)
    
    def normalize(self) -> 'Mesh':
        """Normalize mesh to unit scale centered at origin"""
        if not self.vertices:
            return self
        
        center = self.get_center()
        scale = self.get_scale()
        
        if scale > 1e-8:
            for vertex in self.vertices:
                vertex.position = (vertex.position - center) / scale
        
        self._invalidate_caches()
        return self
    
    def translate(self, translation: Union[List, np.ndarray]) -> 'Mesh':
        """Translate mesh by given vector"""
        translation = np.array(translation, dtype=np.float32)
        for vertex in self.vertices:
            vertex.position += translation
        self._invalidate_caches()
        return self
    
    def rotate(self, rotation_matrix: np.ndarray) -> 'Mesh':
        """Rotate mesh by rotation matrix"""
        rotation_matrix = np.array(rotation_matrix, dtype=np.float32)
        for vertex in self.vertices:
            vertex.position = rotation_matrix @ vertex.position
            if vertex.normal is not None:
                vertex.normal = rotation_matrix @ vertex.normal
        self._invalidate_caches()
        return self
    
    def scale(self, scale_factor: float) -> 'Mesh':
        """Scale mesh by given factor"""
        for vertex in self.vertices:
            vertex.position *= scale_factor
        self._invalidate_caches()
        return self
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get vertex adjacency matrix"""
        if self._adjacency_matrix is not None:
            return self._adjacency_matrix
        
        n_vertices = len(self.vertices)
        adjacency = np.zeros((n_vertices, n_vertices), dtype=np.float32)
        
        for face in self.faces:
            # Connect all vertices in each face
            for i, v1 in enumerate(face.vertex_indices):
                for j, v2 in enumerate(face.vertex_indices):
                    if i != j and v1 < n_vertices and v2 < n_vertices:
                        adjacency[v1, v2] = 1.0
                        adjacency[v2, v1] = 1.0
        
        self._adjacency_matrix = adjacency
        return adjacency
    
    def _invalidate_caches(self):
        """Invalidate cached computations"""
        self._adjacency_matrix = None
        self._vertex_positions_cache = None
        self._face_normals_cache = None
        self._bbox_cache = None
    
    def copy(self) -> 'Mesh':
        """Create a deep copy of the mesh"""
        new_vertices = []
        for v in self.vertices:
            new_v = Vertex(
                v.position.copy(),
                v.normal.copy() if v.normal is not None else None,
                v.color.copy() if v.color is not None else None,
                v.uv.copy() if v.uv is not None else None,
                v.vertex_id
            )
            new_vertices.append(new_v)
        
        new_faces = []
        for f in self.faces:
            new_f = Face(
                f.vertex_indices.copy(),
                f.normal.copy() if f.normal is not None else None,
                f.face_id
            )
            new_faces.append(new_f)
        
        return Mesh(new_vertices, new_faces, self.name)
    
    def __str__(self):
        return f"Mesh('{self.name}', vertices={len(self.vertices)}, faces={len(self.faces)})"
    
    def __repr__(self):
        return self.__str__()

class MeshGenerator:
    """Utility class for generating primitive meshes"""
    
    @staticmethod
    def cube(size: float = 1.0) -> Mesh:
        """Generate a cube mesh"""
        s = size / 2.0
        
        # 8 vertices of a cube
        vertices = [
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # Bottom face
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]       # Top face
        ]
        
        # 12 triangular faces (2 per cube face)
        faces = [
            # Bottom face (z = -s)
            [0, 1, 2], [0, 2, 3],
            # Top face (z = s)
            [4, 6, 5], [4, 7, 6],
            # Front face (y = -s)
            [0, 5, 1], [0, 4, 5],
            # Back face (y = s)
            [2, 6, 7], [2, 7, 3],
            # Left face (x = -s)
            [0, 3, 7], [0, 7, 4],
            # Right face (x = s)
            [1, 5, 6], [1, 6, 2]
        ]
        
        mesh = Mesh(name="Cube")
        for vertex_pos in vertices:
            mesh.add_vertex(vertex_pos)
        for face_indices in faces:
            mesh.add_face(face_indices)
        
        mesh.compute_vertex_normals()
        return mesh
    
    @staticmethod
    def sphere(radius: float = 1.0, subdivisions: int = 2) -> Mesh:
        """Generate a sphere mesh using icosphere subdivision"""
        # Start with icosahedron
        mesh = MeshGenerator._create_icosahedron(radius)
        
        # Subdivide
        for _ in range(subdivisions):
            mesh = MeshGenerator._subdivide_sphere(mesh, radius)
        
        mesh.compute_vertex_normals()
        mesh.name = "Sphere"
        return mesh
    
    @staticmethod
    def _create_icosahedron(radius: float = 1.0) -> Mesh:
        """Create an icosahedron (20-sided polyhedron)"""
        # Golden ratio
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        
        # Icosahedron vertices
        vertices = [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ]
        
        # Normalize to sphere
        vertices = [np.array(v, dtype=np.float32) for v in vertices]
        vertices = [v / np.linalg.norm(v) * radius for v in vertices]
        
        # Icosahedron faces
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]
        
        mesh = Mesh(name="Icosahedron")
        for vertex_pos in vertices:
            mesh.add_vertex(vertex_pos)
        for face_indices in faces:
            mesh.add_face(face_indices)
        
        return mesh
    
    @staticmethod
    def _subdivide_sphere(mesh: Mesh, radius: float) -> Mesh:
        """Subdivide sphere mesh by splitting each triangle"""
        new_mesh = Mesh(name=mesh.name)
        
        # Copy existing vertices
        for vertex in mesh.vertices:
            new_mesh.add_vertex(vertex.position)
        
        # For each face, create 4 new faces
        edge_midpoints = {}  # (v1, v2) -> new_vertex_index
        
        for face in mesh.faces:
            if len(face.vertex_indices) == 3:
                v1, v2, v3 = face.vertex_indices
                
                # Get or create midpoint vertices
                def get_midpoint(va, vb):
                    key = tuple(sorted([va, vb]))
                    if key not in edge_midpoints:
                        pos_a = mesh.vertices[va].position
                        pos_b = mesh.vertices[vb].position
                        midpoint = (pos_a + pos_b) / 2.0
                        # Project to sphere surface
                        midpoint = midpoint / np.linalg.norm(midpoint) * radius
                        edge_midpoints[key] = new_mesh.add_vertex(midpoint)
                    return edge_midpoints[key]
                
                # Get midpoint vertices
                m12 = get_midpoint(v1, v2)
                m23 = get_midpoint(v2, v3)
                m31 = get_midpoint(v3, v1)
                
                # Create 4 new faces
                new_mesh.add_face([v1, m12, m31])
                new_mesh.add_face([v2, m23, m12])
                new_mesh.add_face([v3, m31, m23])
                new_mesh.add_face([m12, m23, m31])
        
        return new_mesh
    
    @staticmethod
    def cylinder(radius: float = 1.0, height: float = 2.0, segments: int = 16) -> Mesh:
        """Generate a cylinder mesh"""
        mesh = Mesh(name="Cylinder")
        
        # Generate vertices
        half_height = height / 2.0
        
        # Bottom center vertex
        bottom_center = mesh.add_vertex([0, 0, -half_height])
        # Top center vertex  
        top_center = mesh.add_vertex([0, 0, half_height])
        
        # Bottom and top ring vertices
        bottom_ring = []
        top_ring = []
        
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            bottom_ring.append(mesh.add_vertex([x, y, -half_height]))
            top_ring.append(mesh.add_vertex([x, y, half_height]))
        
        # Create faces
        # Bottom cap
        for i in range(segments):
            next_i = (i + 1) % segments
            mesh.add_face([bottom_center, bottom_ring[next_i], bottom_ring[i]])
        
        # Top cap
        for i in range(segments):
            next_i = (i + 1) % segments
            mesh.add_face([top_center, top_ring[i], top_ring[next_i]])
        
        # Side faces
        for i in range(segments):
            next_i = (i + 1) % segments
            # Two triangles per side segment
            mesh.add_face([bottom_ring[i], top_ring[i], bottom_ring[next_i]])
            mesh.add_face([bottom_ring[next_i], top_ring[i], top_ring[next_i]])
        
        mesh.compute_vertex_normals()
        return mesh
    
    @staticmethod
    def plane(width: float = 2.0, height: float = 2.0, 
              width_segments: int = 1, height_segments: int = 1) -> Mesh:
        """Generate a plane mesh"""
        mesh = Mesh(name="Plane")
        
        # Generate grid of vertices
        for j in range(height_segments + 1):
            for i in range(width_segments + 1):
                x = (i / width_segments - 0.5) * width
                y = (j / height_segments - 0.5) * height
                mesh.add_vertex([x, y, 0])
        
        # Generate faces
        for j in range(height_segments):
            for i in range(width_segments):
                # Current quad vertices
                v1 = j * (width_segments + 1) + i
                v2 = v1 + 1
                v3 = (j + 1) * (width_segments + 1) + i
                v4 = v3 + 1
                
                # Two triangles per quad
                mesh.add_face([v1, v2, v3])
                mesh.add_face([v2, v4, v3])
        
        mesh.compute_vertex_normals()
        return mesh

class MeshLoader:
    """Utility class for loading mesh files"""
    
    @staticmethod
    def load_obj(filepath: str) -> Mesh:
        """Load mesh from OBJ file"""
        mesh = Mesh(name=os.path.basename(filepath))
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    # Vertex position
                    coords = list(map(float, line.split()[1:4]))
                    mesh.add_vertex(coords)
                elif line.startswith('f '):
                    # Face (convert to 0-based indexing)
                    indices = []
                    for vertex_str in line.split()[1:]:
                        # Handle v/vt/vn format
                        vertex_index = int(vertex_str.split('/')[0]) - 1
                        indices.append(vertex_index)
                    if len(indices) >= 3:
                        mesh.add_face(indices)
        
        mesh.compute_vertex_normals()
        return mesh
    
    @staticmethod
    def save_obj(mesh: Mesh, filepath: str):
        """Save mesh to OBJ file"""
        with open(filepath, 'w') as f:
            # Write vertices
            for vertex in mesh.vertices:
                f.write(f"v {vertex.position[0]} {vertex.position[1]} {vertex.position[2]}\n")
            
            # Write faces (convert to 1-based indexing)
            for face in mesh.faces:
                indices_str = " ".join(str(idx + 1) for idx in face.vertex_indices)
                f.write(f"f {indices_str}\n")

class MeshDataset:
    """Dataset class for collections of meshes"""
    
    def __init__(self, meshes: Optional[List[Mesh]] = None, 
                 labels: Optional[List[int]] = None,
                 mesh_paths: Optional[List[str]] = None):
        self.meshes = meshes or []
        self.labels = labels or []
        self.mesh_paths = mesh_paths or []
        
    def add_mesh(self, mesh: Mesh, label: Optional[int] = None, path: Optional[str] = None):
        """Add a mesh to the dataset"""
        self.meshes.append(mesh)
        if label is not None:
            self.labels.append(label)
        if path is not None:
            self.mesh_paths.append(path)
    
    def load_from_directory(self, directory: str, extensions: List[str] = ['.obj']):
        """Load all meshes from a directory"""
        for filename in os.listdir(directory):
            if any(filename.endswith(ext) for ext in extensions):
                filepath = os.path.join(directory, filename)
                try:
                    if filename.endswith('.obj'):
                        mesh = MeshLoader.load_obj(filepath)
                        self.add_mesh(mesh, path=filepath)
                except Exception as e:
                    print(f"Failed to load {filepath}: {e}")
    
    def __len__(self):
        return len(self.meshes)
    
    def __getitem__(self, idx):
        mesh = self.meshes[idx]
        label = self.labels[idx] if idx < len(self.labels) else None
        path = self.mesh_paths[idx] if idx < len(self.mesh_paths) else None
        return {'mesh': mesh, 'label': label, 'path': path}

# Example usage and testing
def test_mesh_library():
    """Test the mesh library functionality"""
    print("Testing Mesh Library...")
    
    # Test cube generation
    cube = MeshGenerator.cube(size=2.0)
    print(f"Generated cube: {cube}")
    print(f"Bounding box: {cube.get_bounding_box()}")
    
    # Test sphere generation
    sphere = MeshGenerator.sphere(radius=1.5, subdivisions=1)
    print(f"Generated sphere: {sphere}")
    
    # Test cylinder generation
    cylinder = MeshGenerator.cylinder(radius=1.0, height=3.0, segments=8)
    print(f"Generated cylinder: {cylinder}")
    
    # Test mesh operations
    cube_copy = cube.copy()
    cube_copy.translate([1, 0, 0]).scale(0.5)
    print(f"Transformed cube: {cube_copy}")
    
    # Test adjacency matrix
    adjacency = cube.get_adjacency_matrix()
    print(f"Cube adjacency matrix shape: {adjacency.shape}")
    print(f"Cube adjacency connections: {np.sum(adjacency)}")
    
    print("Mesh library test completed!")

if __name__ == "__main__":
    test_mesh_library()