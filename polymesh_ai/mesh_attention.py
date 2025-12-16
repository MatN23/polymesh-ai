# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
import math

class GeometricAttention(nn.Module):
    """Attention mechanism that incorporates geometric relationships"""
    
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1,
                 max_distance: float = 5.0, distance_bins: int = 32):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.max_distance = max_distance
        self.distance_bins = distance_bins
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        # Standard attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Distance-based bias
        self.distance_embedding = nn.Embedding(distance_bins + 1, nhead)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                positions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model] 
            value: [batch_size, seq_len, d_model]
            positions: [batch_size, seq_len, 3] - 3D positions
            mask: [batch_size, seq_len, seq_len] attention mask
        """
        B, L, D = query.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Compute geometric bias
        distance_matrix = torch.cdist(positions, positions, p=2)  # [B, L, L]
        distance_bins = torch.clamp(
            (distance_matrix / self.max_distance * self.distance_bins).long(),
            0, self.distance_bins
        )
        
        # Get distance embeddings and apply to attention
        distance_bias = self.distance_embedding(distance_bins)  # [B, L, L, nhead]
        distance_bias = distance_bias.permute(0, 3, 1, 2)  # [B, nhead, L, L]
        
        scores = scores + distance_bias
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
            scores.masked_fill_(mask == 0, float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)  # [B, nhead, L, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        return self.out_proj(out)

class GraphAttention(nn.Module):
    """Graph attention mechanism for mesh connectivity"""
    
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1,
                 edge_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.edge_dim = edge_dim
        
        # Node projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Edge projections
        self.edge_proj = nn.Linear(edge_dim, nhead)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_nodes, d_model] node features
            edge_index: [2, num_edges] edge connectivity
            edge_attr: [num_edges, edge_dim] edge attributes
        """
        B, N, D = x.shape
        
        Q = self.q_proj(x).view(B, N, self.nhead, self.head_dim)
        K = self.k_proj(x).view(B, N, self.nhead, self.head_dim)
        V = self.v_proj(x).view(B, N, self.nhead, self.head_dim)
        
        # Initialize attention matrix
        attn_matrix = torch.zeros(B, self.nhead, N, N, device=x.device)
        
        # Compute attention only for connected nodes
        src, dst = edge_index[0], edge_index[1]
        
        # Compute attention scores for edges
        q_src = Q[:, src]  # [B, num_edges, nhead, head_dim]
        k_dst = K[:, dst]  # [B, num_edges, nhead, head_dim]
        
        edge_scores = torch.sum(q_src * k_dst, dim=-1) / math.sqrt(self.head_dim)
        
        # Add edge attributes if available
        if edge_attr is not None:
            edge_bias = self.edge_proj(edge_attr)  # [num_edges, nhead]
            edge_scores = edge_scores + edge_bias.unsqueeze(0)
        
        # Fill attention matrix
        for b in range(B):
            attn_matrix[b, :, src, dst] = edge_scores[b].t()
        
        # Apply softmax row-wise (only over connected nodes)
        attn_weights = torch.zeros_like(attn_matrix)
        for b in range(B):
            for h in range(self.nhead):
                for i in range(N):
                    neighbors = dst[src == i]
                    if len(neighbors) > 0:
                        scores = attn_matrix[b, h, i, neighbors]
                        weights = F.softmax(scores, dim=0)
                        attn_weights[b, h, i, neighbors] = weights
        
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        V_reshaped = V.permute(0, 2, 1, 3)  # [B, nhead, N, head_dim]
        out = torch.matmul(attn_weights, V_reshaped)  # [B, nhead, N, head_dim]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        
        return self.out_proj(out)

class MultiScaleAttention(nn.Module):
    """Multi-scale attention for hierarchical mesh processing"""
    
    def __init__(self, d_model: int, scales: List[int] = [1, 2, 4, 8],
                 nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.scales = scales
        self.nhead = nhead
        
        # Separate attention for each scale
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in scales
        ])
        
        # Scale combination
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        self.scale_projection = nn.Linear(d_model * len(scales), d_model)
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            positions: [batch_size, seq_len, 3]
            mask: [batch_size, seq_len]
        """
        B, L, D = x.shape
        scale_outputs = []
        
        for i, (scale, attention) in enumerate(zip(self.scales, self.scale_attentions)):
            # Subsample for different scales
            if scale == 1:
                scale_x = x
                scale_mask = mask
            else:
                # Simple subsampling (can be improved with learned pooling)
                indices = torch.arange(0, L, scale, device=x.device)
                scale_x = x[:, indices]
                scale_mask = mask[:, indices] if mask is not None else None
            
            # Apply attention at this scale
            attn_out, _ = attention(scale_x, scale_x, scale_x, key_padding_mask=scale_mask)
            
            # Upsample back to original resolution if needed
            if scale > 1:
                # Simple repeat upsampling (can be improved)
                upsampled = torch.repeat_interleave(attn_out, scale, dim=1)
                # Truncate to original length
                upsampled = upsampled[:, :L]
                scale_outputs.append(upsampled)
            else:
                scale_outputs.append(attn_out)
        
        # Combine scales
        combined = torch.cat(scale_outputs, dim=-1)  # [B, L, D * num_scales]
        output = self.scale_projection(combined)
        
        return output

class SparseAttention(nn.Module):
    """Sparse attention for large meshes using local neighborhoods"""
    
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1,
                 neighborhood_size: int = 16, sparse_pattern: str = 'local'):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.neighborhood_size = neighborhood_size
        self.sparse_pattern = sparse_pattern
        
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            positions: [batch_size, seq_len, 3]
            adjacency: [batch_size, seq_len, seq_len] adjacency matrix
        """
        B, L, D = x.shape
        output = torch.zeros_like(x)
        
        for b in range(B):
            for i in range(L):
                # Find neighborhood based on pattern
                if self.sparse_pattern == 'local' and adjacency is not None:
                    # Use graph connectivity
                    neighbors = torch.where(adjacency[b, i] > 0)[0]
                elif self.sparse_pattern == 'spatial':
                    # Use spatial distance
                    distances = torch.norm(positions[b, i:i+1] - positions[b], dim=-1)
                    _, neighbors = torch.topk(distances, 
                                            min(self.neighborhood_size, L), 
                                            largest=False)
                else:
                    # Use local window
                    start = max(0, i - self.neighborhood_size // 2)
                    end = min(L, i + self.neighborhood_size // 2)
                    neighbors = torch.arange(start, end, device=x.device)
                
                if len(neighbors) > 0:
                    # Apply attention within neighborhood
                    local_x = x[b:b+1, neighbors]  # [1, neighborhood_size, d_model]
                    local_out, _ = self.attention(local_x, local_x, local_x)
                    
                    # Find position of current token in neighborhood
                    if i in neighbors:
                        local_idx = torch.where(neighbors == i)[0][0]
                        output[b, i] = local_out[0, local_idx]
                    else:
                        output[b, i] = x[b, i]  # No update if not in neighborhood
        
        return output

class MeshTransformerLayer(nn.Module):
    """Enhanced transformer layer with mesh-aware attention"""
    
    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.1, attention_type: str = 'geometric'):
        super().__init__()
        self.attention_type = attention_type
        
        # Choose attention mechanism
        if attention_type == 'geometric':
            self.self_attn = GeometricAttention(d_model, nhead, dropout)
        elif attention_type == 'graph':
            self.self_attn = GraphAttention(d_model, nhead, dropout)
        elif attention_type == 'multiscale':
            self.self_attn = MultiScaleAttention(d_model, [1, 2, 4], nhead, dropout)
        elif attention_type == 'sparse':
            self.self_attn = SparseAttention(d_model, nhead, dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                adjacency: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            positions: [batch_size, seq_len, 3]
            mask: attention mask
            adjacency: adjacency matrix for graph attention
            edge_index: edge indices for graph attention
        """
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        
        if self.attention_type == 'geometric':
            attn_out = self.self_attn(x, x, x, positions, mask)
        elif self.attention_type == 'graph':
            if edge_index is not None:
                attn_out = self.self_attn(x, edge_index)
            else:
                # Fallback to standard attention
                attn_out, _ = nn.MultiheadAttention(
                    x.shape[-1], 8, batch_first=True
                ).to(x.device)(x, x, x, key_padding_mask=mask)
        elif self.attention_type in ['multiscale', 'sparse']:
            attn_out = self.self_attn(x, positions, mask)
        else:
            attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        
        x = residual + self.dropout(attn_out)
        
        # Feedforward with residual connection
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        
        return x

class AdaptiveMeshTransformer(nn.Module):
    """Adaptive transformer that can switch attention mechanisms dynamically"""
    
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Multiple attention types
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'geometric': MeshTransformerLayer(d_model, nhead, dim_feedforward, 
                                                dropout, 'geometric'),
                'graph': MeshTransformerLayer(d_model, nhead, dim_feedforward, 
                                            dropout, 'graph'),
                'multiscale': MeshTransformerLayer(d_model, nhead, dim_feedforward, 
                                                 dropout, 'multiscale'),
                'sparse': MeshTransformerLayer(d_model, nhead, dim_feedforward, 
                                             dropout, 'sparse')
            })
            for _ in range(num_layers)
        ])
        
        # Attention selection network
        self.attention_selector = nn.Sequential(
            nn.Linear(d_model + 3 + 1, 64),  # features + position + density
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 attention types
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                adjacency: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            positions: [batch_size, seq_len, 3]
            adjacency: [batch_size, seq_len, seq_len]
            edge_index: [2, num_edges]
            mask: [batch_size, seq_len]
        """
        B, L, D = x.shape
        
        for layer_dict in self.layers:
            # Compute local density as a feature for attention selection
            if adjacency is not None:
                density = adjacency.sum(dim=-1, keepdim=True).float()  # [B, L, 1]
            else:
                density = torch.ones(B, L, 1, device=x.device)
            
            # Features for attention selection
            selector_input = torch.cat([
                x.mean(dim=-1, keepdim=True),  # Summarized features
                positions,  # Position
                density  # Local density
            ], dim=-1)  # [B, L, d_model/d_model + 3 + 1]
            
            # Actually, let's use a simpler approach - global average
            global_features = torch.cat([
                x.mean(dim=1),  # [B, d_model]
                positions.mean(dim=1),  # [B, 3]
                density.mean(dim=1)  # [B, 1]
            ], dim=-1)  # [B, d_model + 4]
            
            # Select attention weights
            attn_weights = self.attention_selector(global_features)  # [B, 4]
            
            # Apply weighted combination of attention mechanisms
            layer_outputs = []
            attention_types = ['geometric', 'graph', 'multiscale', 'sparse']
            
            for i, attn_type in enumerate(attention_types):
                layer = layer_dict[attn_type]
                if attn_type == 'graph' and edge_index is not None:
                    out = layer(x, positions, mask, adjacency, edge_index)
                else:
                    out = layer(x, positions, mask, adjacency)
                layer_outputs.append(out)
            
            # Weighted combination
            x = torch.zeros_like(x)
            for i, output in enumerate(layer_outputs):
                weight = attn_weights[:, i:i+1, None]  # [B, 1, 1]
                x = x + weight * output
        
        return x

class MeshTransformerPreTrainer:
    """Pre-training utilities for mesh transformers"""
    
    def __init__(self, model: nn.Module, tokenizer, device: torch.device = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def masked_mesh_modeling(self, tokens: List, mask_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Masked mesh modeling pre-training task"""
        seq_len = len(tokens)
        
        # Create random mask
        mask = torch.rand(seq_len) < mask_ratio
        
        # Extract features and positions
        features = torch.tensor([token.features for token in tokens], dtype=torch.float32, device=self.device)
        positions = torch.tensor([token.position for token in tokens], dtype=torch.float32, device=self.device)
        
        # Create masked input
        masked_features = features.clone()
        masked_features[mask] = 0  # Zero out masked tokens
        
        return masked_features.unsqueeze(0), positions.unsqueeze(0), features[mask]
    
    def mesh_completion_task(self, partial_tokens: List, complete_tokens: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mesh completion pre-training task"""
        # Convert tokens to tensors
        partial_features = torch.tensor([token.features for token in partial_tokens], 
                                      dtype=torch.float32, device=self.device)
        complete_features = torch.tensor([token.features for token in complete_tokens], 
                                       dtype=torch.float32, device=self.device)
        
        return partial_features.unsqueeze(0), complete_features.unsqueeze(0)
    
    def next_token_prediction(self, tokens: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """Next token prediction task for autoregressive training"""
        features = torch.tensor([token.features for token in tokens], dtype=torch.float32, device=self.device)
        
        # Input is all tokens except last, target is all tokens except first
        input_features = features[:-1].unsqueeze(0)
        target_features = features[1:].unsqueeze(0)
        
        return input_features, target_features

# Example usage
def example_advanced_usage():
    """Demonstrate advanced mesh transformer features"""
    
    # Create adaptive transformer
    model = AdaptiveMeshTransformer(d_model=256, nhead=8, num_layers=6)
    
    # Dummy data
    batch_size, seq_len = 2, 100
    x = torch.randn(batch_size, seq_len, 256)
    positions = torch.randn(batch_size, seq_len, 3)
    
    # Create dummy adjacency matrix (sparse)
    adjacency = torch.zeros(batch_size, seq_len, seq_len)
    for b in range(batch_size):
        for i in range(seq_len):
            # Connect to 5 random neighbors
            neighbors = torch.randint(0, seq_len, (5,))
            adjacency[b, i, neighbors] = 1
            adjacency[b, neighbors, i] = 1  # Make symmetric
    
    print("Adaptive Mesh Transformer:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x, positions, adjacency)
        print(f"Output shape: {output.shape}")
    
    # Test different attention mechanisms
    attention_types = ['geometric', 'graph', 'multiscale', 'sparse']
    for attn_type in attention_types:
        layer = MeshTransformerLayer(256, 8, attention_type=attn_type)
        with torch.no_grad():
            if attn_type == 'graph':
                # Create dummy edge index
                edge_index = torch.randint(0, seq_len, (2, seq_len * 3))
                out = layer(x, positions, edge_index=edge_index)
            else:
                out = layer(x, positions, adjacency=adjacency)
            print(f"{attn_type} attention output shape: {out.shape}")

if __name__ == "__main__":
    example_advanced_usage()