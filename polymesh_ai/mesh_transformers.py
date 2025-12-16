# mesh_transformers.py
# Copyright (c) 2025 Matias Nielsen. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import math

class MeshToken:
    """Represents a single mesh token (vertex, face, or patch)"""
    
    def __init__(self, token_id: int, features: np.ndarray, 
                 position: np.ndarray, token_type: str = 'vertex',
                 metadata: Optional[Dict] = None):
        self.token_id = token_id
        self.features = features  # Geometric features (normals, curvature, etc.)
        self.position = position  # 3D position
        self.token_type = token_type
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"MeshToken(id={self.token_id}, type={self.token_type}, pos={self.position})"

class VertexTokenizer:
    """Tokenizes mesh by vertices"""
    
    def __init__(self, include_normals: bool = True, include_colors: bool = False):
        self.include_normals = include_normals
        self.include_colors = include_colors
    
    def tokenize(self, mesh) -> List[MeshToken]:
        tokens = []
        
        for i, vertex in enumerate(mesh.vertices):
            # Base features: position
            features = vertex.position.copy()
            
            # Add normal if available and requested
            if self.include_normals and vertex.normal is not None:
                features = np.concatenate([features, vertex.normal])
            elif self.include_normals:
                # Use zero normal if not available
                features = np.concatenate([features, np.zeros(3)])
            
            # Add color if available and requested
            if self.include_colors and vertex.color is not None:
                features = np.concatenate([features, vertex.color])
            elif self.include_colors:
                features = np.concatenate([features, np.zeros(3)])
            
            token = MeshToken(
                token_id=i,
                features=features,
                position=vertex.position,
                token_type='vertex'
            )
            tokens.append(token)
        
        return tokens

class FaceTokenizer:
    """Tokenizes mesh by faces"""
    
    def __init__(self, max_face_vertices: int = 4):
        self.max_face_vertices = max_face_vertices
    
    def tokenize(self, mesh) -> List[MeshToken]:
        tokens = []
        
        for i, face in enumerate(mesh.faces):
            # Get face center
            face_positions = [mesh.vertices[idx].position for idx in face.vertex_indices]
            center = np.mean(face_positions, axis=0)
            
            # Compute face features
            if face.normal is not None:
                normal = face.normal
            else:
                # Compute face normal
                if len(face_positions) >= 3:
                    v1 = face_positions[1] - face_positions[0]
                    v2 = face_positions[2] - face_positions[0]
                    normal = np.cross(v1, v2)
                    norm = np.linalg.norm(normal)
                    if norm > 1e-8:
                        normal = normal / norm
                    else:
                        normal = np.array([0, 0, 1])
                else:
                    normal = np.array([0, 0, 1])
            
            # Face area
            if len(face_positions) >= 3:
                v1 = face_positions[1] - face_positions[0]
                v2 = face_positions[2] - face_positions[0]
                area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            else:
                area = 0.0
            
            # Features: center + normal + area + vertex count
            features = np.concatenate([center, normal, [area], [len(face.vertex_indices)]])
            
            token = MeshToken(
                token_id=i,
                features=features,
                position=center,
                token_type='face',
                metadata={'vertex_indices': face.vertex_indices}
            )
            tokens.append(token)
        
        return tokens

class PatchTokenizer:
    """Tokenizes mesh by patches"""
    
    def __init__(self, patch_size: int = 16):
        self.patch_size = patch_size
    
    def tokenize(self, mesh) -> List[MeshToken]:
        # This is a simplified patch tokenization
        # In practice, you'd use more sophisticated clustering
        tokens = []
        
        vertices_per_patch = max(1, len(mesh.vertices) // self.patch_size)
        
        for patch_id in range(self.patch_size):
            start_idx = patch_id * vertices_per_patch
            end_idx = min((patch_id + 1) * vertices_per_patch, len(mesh.vertices))
            
            if start_idx >= len(mesh.vertices):
                break
            
            # Get vertices in this patch
            patch_vertices = mesh.vertices[start_idx:end_idx]
            
            # Compute patch center and features
            positions = [v.position for v in patch_vertices]
            center = np.mean(positions, axis=0)
            
            # Patch features: center + size + bounding box diagonal
            bbox_min = np.min(positions, axis=0)
            bbox_max = np.max(positions, axis=0)
            bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
            
            features = np.concatenate([center, [len(patch_vertices)], [bbox_diagonal]])
            
            token = MeshToken(
                token_id=patch_id,
                features=features,
                position=center,
                token_type='patch',
                metadata={'vertex_range': (start_idx, end_idx)}
            )
            tokens.append(token)
        
        return tokens

class PositionalEncoding(nn.Module):
    """3D positional encoding for mesh tokens"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create learnable positional embeddings for 3D coordinates
        self.pos_embedding = nn.Linear(3, d_model)
        
        # Optional: Add sinusoidal encoding
        self.use_sinusoidal = True
        if self.use_sinusoidal:
            self.register_buffer('freq_bands', torch.logspace(0, 10, d_model // 6))
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [batch_size, seq_len, 3] 3D positions
        Returns:
            pos_encoding: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = positions.shape
        
        if self.use_sinusoidal:
            # Sinusoidal encoding for 3D positions
            pos_enc_list = []
            
            for i in range(3):  # x, y, z coordinates
                coord = positions[..., i:i+1]  # [batch_size, seq_len, 1]
                
                # Apply frequency bands
                coord_scaled = coord.unsqueeze(-1) * self.freq_bands  # [batch_size, seq_len, 1, d_model//6]
                
                # Sin and cos
                sin_enc = torch.sin(coord_scaled).squeeze(-2)  # [batch_size, seq_len, d_model//6]
                cos_enc = torch.cos(coord_scaled).squeeze(-2)  # [batch_size, seq_len, d_model//6]
                
                pos_enc_list.extend([sin_enc, cos_enc])
            
            pos_encoding = torch.cat(pos_enc_list, dim=-1)  # [batch_size, seq_len, d_model]
            
            # Ensure correct dimension
            if pos_encoding.shape[-1] > self.d_model:
                pos_encoding = pos_encoding[..., :self.d_model]
            elif pos_encoding.shape[-1] < self.d_model:
                padding = torch.zeros(batch_size, seq_len, self.d_model - pos_encoding.shape[-1], 
                                    device=positions.device)
                pos_encoding = torch.cat([pos_encoding, padding], dim=-1)
        else:
            # Simple learned embedding
            pos_encoding = self.pos_embedding(positions)
        
        return pos_encoding

class GeometricSelfAttention(nn.Module):
    """Self-attention with geometric bias for 3D meshes"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert self.head_dim * nhead == d_model
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Geometric bias networks
        self.distance_bias = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, nhead)
        )
        
        self.angle_bias = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, nhead)
        )
        
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] input features
            positions: [batch_size, seq_len, 3] 3D positions
            mask: [batch_size, seq_len] attention mask
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, nhead, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Compute geometric biases
        # Distance bias
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [batch_size, seq_len, seq_len, 3]
        distances = torch.norm(pos_diff, dim=-1, keepdim=True)  # [batch_size, seq_len, seq_len, 1]
        distance_bias = self.distance_bias(distances)  # [batch_size, seq_len, seq_len, nhead]
        distance_bias = distance_bias.permute(0, 3, 1, 2)  # [batch_size, nhead, seq_len, seq_len]
        
        # Add geometric bias to attention scores
        scores = scores + distance_bias
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            mask_expanded = mask_expanded.expand(-1, self.nhead, seq_len, -1)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        out = torch.matmul(attention_weights, v)  # [batch_size, nhead, seq_len, head_dim]
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        out = self.output_linear(out)
        
        return out

class MeshTransformerLayer(nn.Module):
    """Single transformer layer for mesh processing"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = GeometricSelfAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.self_attn(x, positions, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class MeshTransformer(nn.Module):
    """Complete mesh transformer model"""
    
    def __init__(self, feature_dim: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, num_classes: int = 10):
        super().__init__()
        
        self.d_model = d_model
        self.feature_dim = feature_dim
        
        # Input projection
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MeshTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Task-specific heads
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, feature_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tokens: List[MeshToken], task: str = 'classification') -> torch.Tensor:
        """
        Args:
            tokens: List of MeshToken objects
            task: 'classification' or 'reconstruction'
        """
        if len(tokens) == 0:
            if task == 'classification':
                return torch.zeros(1, 10)  # num_classes
            else:
                return torch.zeros(1, self.feature_dim)
        
        # Convert tokens to tensors
        features = torch.tensor([token.features for token in tokens], dtype=torch.float32)
        positions = torch.tensor([token.position for token in tokens], dtype=torch.float32)
        
        # Add batch dimension
        features = features.unsqueeze(0)  # [1, seq_len, feature_dim]
        positions = positions.unsqueeze(0)  # [1, seq_len, 3]
        
        # Input projection
        x = self.input_projection(features)  # [1, seq_len, d_model]
        
        # Add positional encoding
        pos_enc = self.pos_encoding(positions)
        x = x + pos_enc
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, positions)
        
        # Task-specific output
        if task == 'classification':
            # Global average pooling
            x = x.mean(dim=1)  # [1, d_model]
            output = self.classification_head(x)
        elif task == 'reconstruction':
            # Per-token reconstruction
            output = self.reconstruction_head(x)  # [1, seq_len, feature_dim]
        else:
            raise ValueError(f"Unknown task: {task}")
        
        return output

class AdaptiveMeshTransformer(nn.Module):
    """Adaptive mesh transformer that can handle varying input sizes"""
    
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, 
                 num_classes: int = 10):
        super().__init__()
        
        self.d_model = d_model
        
        # Adaptive input projection (handles variable feature dimensions)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MeshTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Adaptive pooling and classification
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features: torch.Tensor, positions: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [batch_size, seq_len, feature_dim] mesh features
            positions: [batch_size, seq_len, 3] 3D positions  
            mask: [batch_size, seq_len] attention mask
        """
        # Project to model dimension if needed
        if features.shape[-1] != self.d_model:
            # Adaptive projection
            feature_proj = nn.Linear(features.shape[-1], self.d_model).to(features.device)
            x = feature_proj(features)
        else:
            x = features
        
        x = self.input_norm(x)
        
        # Add positional encoding
        pos_enc = self.pos_encoding(positions)
        x = x + pos_enc
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, positions, mask)
        
        # Global pooling
        if mask is not None:
            # Masked average pooling
            x = x * mask.unsqueeze(-1)
            x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            # Standard average pooling
            x = x.mean(dim=1)
        
        # Classification
        output = self.classification_head(x)
        
        return output

# Factory functions for easy model creation
def create_mesh_classifier(feature_dim: int = 6, num_classes: int = 10, 
                          d_model: int = 256, nhead: int = 8, 
                          num_layers: int = 4) -> MeshTransformer:
    """Create a mesh classification model"""
    return MeshTransformer(
        feature_dim=feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes
    )

def create_mesh_autoencoder(feature_dim: int = 6, d_model: int = 512,
                           nhead: int = 8, num_layers: int = 6) -> MeshTransformer:
    """Create a mesh autoencoder model"""
    return MeshTransformer(
        feature_dim=feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=feature_dim  # Reconstruct input features
    )

def create_adaptive_classifier(num_classes: int = 10, d_model: int = 256,
                              nhead: int = 8, num_layers: int = 4) -> AdaptiveMeshTransformer:
    """Create an adaptive mesh classifier"""
    return AdaptiveMeshTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes
    )