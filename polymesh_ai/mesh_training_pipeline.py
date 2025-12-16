# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import json
import os
from tqdm import tqdm
import wandb
from collections import defaultdict
import pickle

from . import mesh_transformers
from . import mesh_attention
from .mesh_transformers import MeshToken, VertexTokenizer, FaceTokenizer, PatchTokenizer
from .mesh_attention import AdaptiveMeshTransformer

class MeshTransformerDataset(Dataset):
    """Dataset class for mesh transformer training"""
    
    def __init__(self, mesh_data: List[Dict], tokenizer, 
                 task_type: str = 'classification',
                 augmentation_fn: Optional[Callable] = None,
                 max_seq_len: int = 1024):
        self.mesh_data = mesh_data
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.augmentation_fn = augmentation_fn
        self.max_seq_len = max_seq_len
        
        # Pre-tokenize all meshes for efficiency
        self.tokenized_data = []
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Pre-tokenize all meshes"""
        print("Pre-tokenizing meshes...")
        for item in tqdm(self.mesh_data):
            mesh = item['mesh']
            
            # Apply augmentation if provided
            if self.augmentation_fn:
                mesh = self.augmentation_fn(mesh)
            
            # Tokenize mesh
            tokens = self.tokenizer.tokenize(mesh)
            
            # Truncate if too long
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            
            # Store tokenized data
            processed_item = {
                'tokens': tokens,
                'label': item.get('label', 0),
                'mesh_id': item.get('id', ''),
                'metadata': item.get('metadata', {})
            }
            self.tokenized_data.append(processed_item)
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        batch_tokens = [item['tokens'] for item in batch]
        batch_labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        
        # Get max sequence length in batch
        max_len = max(len(tokens) for tokens in batch_tokens)
        
        # Pad sequences and create attention masks
        padded_features = []
        padded_positions = []
        attention_masks = []
        
        for tokens in batch_tokens:
            seq_len = len(tokens)
            
            # Extract features and positions
            features = np.array([token.features for token in tokens])
            positions = np.array([token.position for token in tokens])
            
            # Pad sequences
            feature_dim = features.shape[1] if len(features) > 0 else 3
            position_dim = positions.shape[1] if len(positions) > 0 else 3
            
            padded_feat = np.zeros((max_len, feature_dim), dtype=np.float32)
            padded_pos = np.zeros((max_len, position_dim), dtype=np.float32)
            mask = np.zeros(max_len, dtype=bool)
            
            if seq_len > 0:
                padded_feat[:seq_len] = features
                padded_pos[:seq_len] = positions
                mask[:seq_len] = True
            
            padded_features.append(padded_feat)
            padded_positions.append(padded_pos)
            attention_masks.append(mask)
        
        return {
            'features': torch.tensor(np.array(padded_features), dtype=torch.float32),
            'positions': torch.tensor(np.array(padded_positions), dtype=torch.float32),
            'attention_mask': torch.tensor(np.array(attention_masks), dtype=torch.bool),
            'labels': batch_labels,
            'batch_tokens': batch_tokens  # Keep original tokens for reference
        }

class MeshTransformerTrainingPipeline:
    """Complete training pipeline for mesh transformers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize model based on config
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Loss function
        self.criterion = self._build_criterion()
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        self.best_metric = float('-inf') if config.get('maximize_metric', True) else float('inf')
        
        # Initialize tokenizer
        self.tokenizer = self._build_tokenizer()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        
    def _build_model(self) -> nn.Module:
        """Build model based on config"""
        model_type = self.config.get('model_type', 'standard')
        
        if model_type == 'adaptive':
            model = AdaptiveMeshTransformer(
                d_model=self.config.get('d_model', 512),
                nhead=self.config.get('nhead', 8),
                num_layers=self.config.get('num_layers', 6),
                dim_feedforward=self.config.get('dim_feedforward', 2048),
                dropout=self.config.get('dropout', 0.1)
            )
        else:
            from mesh_transformers import MeshTransformer
            model = MeshTransformer(
                feature_dim=self.config.get('feature_dim', 6),
                d_model=self.config.get('d_model', 512),
                nhead=self.config.get('nhead', 8),
                num_layers=self.config.get('num_layers', 6),
                dim_feedforward=self.config.get('dim_feedforward', 2048),
                dropout=self.config.get('dropout', 0.1)
            )
        
        return model
    
    def _build_tokenizer(self):
        """Build tokenizer based on config"""
        tokenizer_type = self.config.get('tokenizer_type', 'vertex')
        
        if tokenizer_type == 'vertex':
            from mesh_transformers import VertexTokenizer
            return VertexTokenizer(
                include_normals=self.config.get('include_normals', True),
                include_colors=self.config.get('include_colors', False)
            )
        elif tokenizer_type == 'face':
            from mesh_transformers import FaceTokenizer
            return FaceTokenizer(
                max_face_vertices=self.config.get('max_face_vertices', 4)
            )
        elif tokenizer_type == 'patch':
            from mesh_transformers import PatchTokenizer
            return PatchTokenizer(
                patch_size=self.config.get('patch_size', 16)
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        if optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.get('max_epochs', 100)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.get('step_size', 30), gamma=0.1
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max' if self.config.get('maximize_metric', True) else 'min'
            )
        else:
            return None
    
    def _build_criterion(self):
        """Build loss function"""
        task_type = self.config.get('task_type', 'classification')
        
        if task_type == 'classification':
            return nn.CrossEntropyLoss()
        elif task_type == 'regression':
            return nn.MSELoss()
        elif task_type == 'reconstruction':
            return nn.MSELoss()
        else:
            return nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # Move batch to device
            features = batch['features'].to(self.device)
            positions = batch['positions'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass through embedding and model
            batch_size, seq_len, feature_dim = features.shape
            
            # Convert features and positions to token format for the model
            batch_outputs = []
            for b in range(batch_size):
                # Get valid (non-padded) tokens
                valid_mask = attention_mask[b]
                valid_features = features[b, valid_mask]
                valid_positions = positions[b, valid_mask]
                
                # Create token objects (simplified for this example)
                from mesh_transformers import MeshToken
                tokens = []
                for i in range(len(valid_features)):
                    token = MeshToken(
                        token_id=i,
                        features=valid_features[i].cpu().numpy(),
                        position=valid_positions[i].cpu().numpy(),
                        token_type='vertex'
                    )
                    tokens.append(token)
                
                # Forward pass for this sample
                if isinstance(self.model, AdaptiveMeshTransformer):
                    # For adaptive model, we need tensor input
                    sample_features = valid_features.unsqueeze(0)
                    sample_positions = valid_positions.unsqueeze(0)
                    output = self.model(sample_features, sample_positions)
                    # Global average pooling for classification
                    output = output.mean(dim=1)  # [1, d_model]
                else:
                    # For standard mesh transformer
                    output = self.model(tokens, task='classification')
                
                batch_outputs.append(output)
            
            # Stack outputs
            if batch_outputs:
                outputs = torch.cat(batch_outputs, dim=0)
            else:
                continue
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Update metrics
            epoch_metrics['loss'].append(loss.item())
            
            # Compute accuracy for classification
            if self.config.get('task_type', 'classification') == 'classification':
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == labels).float().mean().item()
                epoch_metrics['accuracy'].append(acc)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{epoch_metrics['accuracy'][-1]:.4f}" if epoch_metrics['accuracy'] else "N/A"
            })
            
            self.global_step += 1
            
            # Log to wandb if configured
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train/loss': loss.item(),
                    'train/accuracy': epoch_metrics['accuracy'][-1] if epoch_metrics['accuracy'] else 0,
                    'global_step': self.global_step
                })
        
        # Average metrics over epoch
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        return avg_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move batch to device
                features = batch['features'].to(self.device)
                positions = batch['positions'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass (same as training)
                batch_size, seq_len, feature_dim = features.shape
                batch_outputs = []
                
                for b in range(batch_size):
                    valid_mask = attention_mask[b]
                    valid_features = features[b, valid_mask]
                    valid_positions = positions[b, valid_mask]
                    
                    from mesh_transformers import MeshToken
                    tokens = []
                    for i in range(len(valid_features)):
                        token = MeshToken(
                            token_id=i,
                            features=valid_features[i].cpu().numpy(),
                            position=valid_positions[i].cpu().numpy(),
                            token_type='vertex'
                        )
                        tokens.append(token)
                    
                    if isinstance(self.model, AdaptiveMeshTransformer):
                        sample_features = valid_features.unsqueeze(0)
                        sample_positions = valid_positions.unsqueeze(0)
                        output = self.model(sample_features, sample_positions)
                        output = output.mean(dim=1)
                    else:
                        output = self.model(tokens, task='classification')
                    
                    batch_outputs.append(output)
                
                if batch_outputs:
                    outputs = torch.cat(batch_outputs, dim=0)
                else:
                    continue
                
                # Compute metrics
                loss = self.criterion(outputs, labels)
                val_metrics['loss'].append(loss.item())
                
                if self.config.get('task_type', 'classification') == 'classification':
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == labels).float().mean().item()
                    val_metrics['accuracy'].append(acc)
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        return avg_metrics
    
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        """Main training loop"""
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            collate_fn=MeshTransformerDataset.collate_fn,
            num_workers=self.config.get('num_workers', 4)
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=False,
                collate_fn=MeshTransformerDataset.collate_fn,
                num_workers=self.config.get('num_workers', 4)
            )
        
        # Initialize wandb if configured
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'mesh-transformer'),
                config=self.config
            )
        
        max_epochs = self.config.get('max_epochs', 100)
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        patience_counter = 0
        
        for epoch in range(max_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                print(f"Epoch {epoch}: Val Loss: {val_metrics['loss']:.4f}")
                
                # Check for improvement
                current_metric = val_metrics.get('accuracy', val_metrics['loss'])
                if self.config.get('maximize_metric', True):
                    improved = current_metric > self.best_metric
                else:
                    improved = current_metric < self.best_metric
                
                if improved:
                    self.best_metric = current_metric
                    patience_counter = 0
                    self.save_checkpoint('best_model.pth')
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                # Log validation metrics
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'val/loss': val_metrics['loss'],
                        'val/accuracy': val_metrics.get('accuracy', 0),
                        'epoch': epoch
                    })
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    metric = val_metrics.get('accuracy', val_metrics['loss']) if val_loader else train_metrics['loss']
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        if self.config.get('use_wandb', False):
            wandb.finish()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        os.makedirs(self.config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
        filepath = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        print(f"Checkpoint loaded from {filepath}")

class MeshAugmentation:
    """Data augmentation utilities for meshes"""
    
    @staticmethod
    def random_rotation(mesh, angle_range: float = 30.0):
        """Apply random rotation"""
        angle = np.random.uniform(-angle_range, angle_range) * np.pi / 180
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # Rodrigues rotation formula
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = (cos_angle * np.eye(3) + 
                          sin_angle * np.array([[0, -axis[2], axis[1]],
                                              [axis[2], 0, -axis[0]],
                                              [-axis[1], axis[0], 0]]) +
                          (1 - cos_angle) * np.outer(axis, axis))
        
        return mesh.rotate(rotation_matrix)
    
    @staticmethod
    def random_scale(mesh, scale_range: Tuple[float, float] = (0.8, 1.2)):
        """Apply random scaling"""
        scale = np.random.uniform(*scale_range)
        return mesh.scale(scale)
    
    @staticmethod
    def random_translation(mesh, translation_range: float = 0.1):
        """Apply random translation"""
        translation = np.random.uniform(-translation_range, translation_range, 3)
        return mesh.translate(translation)
    
    @staticmethod
    def add_noise(mesh, noise_std: float = 0.01):
        """Add random noise to vertices"""
        for vertex in mesh.vertices:
            noise = np.random.normal(0, noise_std, 3)
            vertex.position += noise
        mesh._invalidate_caches()
        return mesh
    
    @staticmethod
    def random_augment(mesh):
        """Apply random combination of augmentations"""
        # Copy mesh to avoid modifying original
        import copy
        augmented_mesh = copy.deepcopy(mesh)
        
        # Apply random augmentations
        if np.random.rand() > 0.5:
            augmented_mesh = MeshAugmentation.random_rotation(augmented_mesh)
        if np.random.rand() > 0.5:
            augmented_mesh = MeshAugmentation.random_scale(augmented_mesh)
        if np.random.rand() > 0.5:
            augmented_mesh = MeshAugmentation.random_translation(augmented_mesh)
        if np.random.rand() > 0.3:
            augmented_mesh = MeshAugmentation.add_noise(augmented_mesh)
        
        return augmented_mesh

class MeshDatasetBuilder:
    """Utility class for building mesh datasets"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def build_classification_dataset(self, class_folders: List[str]) -> List[Dict]:
        """Build classification dataset from folder structure"""
        dataset = []
        
        for class_idx, class_folder in enumerate(class_folders):
            class_path = os.path.join(self.data_dir, class_folder)
            if not os.path.exists(class_path):
                continue
            
            mesh_files = [f for f in os.listdir(class_path) 
                         if f.endswith(('.obj', '.ply', '.pkl'))]
            
            for mesh_file in mesh_files:
                mesh_path = os.path.join(class_path, mesh_file)
                dataset.append({
                    'mesh_path': mesh_path,
                    'label': class_idx,
                    'class_name': class_folder,
                    'id': f"{class_folder}_{mesh_file}"
                })
        
        return dataset
    
    def build_segmentation_dataset(self, mesh_files: List[str], 
                                 label_files: List[str]) -> List[Dict]:
        """Build segmentation dataset"""
        dataset = []
        
        for mesh_file, label_file in zip(mesh_files, label_files):
            dataset.append({
                'mesh_path': mesh_file,
                'labels_path': label_file,
                'task_type': 'segmentation',
                'id': os.path.basename(mesh_file)
            })
        
        return dataset

# Example training script
def train_mesh_classifier():
    """Example training script for mesh classification"""
    
    # Configuration
    config = {
        'model_type': 'adaptive',  # or 'standard'
        'tokenizer_type': 'vertex',
        'feature_dim': 6,  # 3 position + 3 normal
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 16,
        'max_epochs': 100,
        'early_stopping_patience': 15,
        'grad_clip': 1.0,
        'device': 'cuda',
        'use_wandb': False,
        'checkpoint_dir': 'checkpoints',
        'include_normals': True,
        'include_colors': False,
        'max_seq_len': 512,
        'task_type': 'classification'
    }
    
    # Initialize pipeline
    pipeline = MeshTransformerTrainingPipeline(config)
    
    # Create dummy dataset (in practice, load from files)
    from mesh_library import MeshGenerator
    
    # Generate dummy meshes for different classes
    train_data = []
    val_data = []
    
    mesh_types = ['cube', 'sphere', 'cylinder']
    for i, mesh_type in enumerate(mesh_types):
        for j in range(100):  # 100 samples per class
            if mesh_type == 'cube':
                mesh = MeshGenerator.cube(size=np.random.uniform(0.5, 2.0))
            elif mesh_type == 'sphere':
                mesh = MeshGenerator.sphere(radius=np.random.uniform(0.5, 1.5), 
                                          subdivisions=np.random.randint(1, 3))
            else:  # cylinder
                mesh = MeshGenerator.cylinder(radius=np.random.uniform(0.5, 1.2),
                                            height=np.random.uniform(1.0, 3.0))
            
            # Apply normalization and compute normals
            mesh.normalize().compute_vertex_normals()
            
            data_point = {
                'mesh': mesh,
                'label': i,
                'id': f"{mesh_type}_{j}",
                'metadata': {'mesh_type': mesh_type}
            }
            
            if j < 80:  # 80% for training
                train_data.append(data_point)
            else:  # 20% for validation
                val_data.append(data_point)
    
    # Create datasets
    train_dataset = MeshTransformerDataset(
        train_data, 
        pipeline.tokenizer,
        task_type='classification',
        augmentation_fn=MeshAugmentation.random_augment,
        max_seq_len=config['max_seq_len']
    )
    
    val_dataset = MeshTransformerDataset(
        val_data,
        pipeline.tokenizer,
        task_type='classification',
        max_seq_len=config['max_seq_len']
    )
    
    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    print(f"Model parameters: {sum(p.numel() for p in pipeline.model.parameters()):,}")
    
    # Start training
    pipeline.train(train_dataset, val_dataset)
    
    return pipeline

def train_mesh_autoencoder():
    """Example training script for mesh autoencoder"""
    
    config = {
        'model_type': 'standard',
        'tokenizer_type': 'vertex',
        'feature_dim': 6,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 8,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'batch_size': 8,
        'max_epochs': 200,
        'early_stopping_patience': 20,
        'grad_clip': 1.0,
        'device': 'cuda',
        'task_type': 'reconstruction',
        'max_seq_len': 1024,
        'checkpoint_dir': 'checkpoints_autoencoder'
    }
    
    pipeline = MeshTransformerTrainingPipeline(config)
    print("Mesh Autoencoder Pipeline initialized")
    print(f"Model parameters: {sum(p.numel() for p in pipeline.model.parameters()):,}")
    
    return pipeline

# Main execution
if __name__ == "__main__":
    print("Mesh Transformer Training Pipeline")
    print("="*50)
    
    # Example 1: Classification
    print("\n1. Training Mesh Classifier...")
    classifier_pipeline = train_mesh_classifier()
    
    # Example 2: Autoencoder setup
    print("\n2. Setting up Mesh Autoencoder...")
    autoencoder_pipeline = train_mesh_autoencoder()
    
    print("\nTraining pipeline examples completed!")